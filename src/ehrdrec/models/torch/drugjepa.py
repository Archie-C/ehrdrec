import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JEPAPredictor(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        return self.net(z_context)


class DrugSetTargetEncoder(nn.Module):
    """
    Encodes a multi-hot medication set into a structured latent target.

    Unlike mean pooling, this lets drugs interact with each other before pooling,
    so the target can represent combinations, not just an average centroid.

    Input:
        multihot: [B, num_meds]

    Output:
        z_target: [B, embedding_dim]
    """

    def __init__(
        self,
        num_meds: int,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_set_size: int = 64,
    ) -> None:
        super().__init__()

        self.num_meds = num_meds
        self.embedding_dim = embedding_dim
        self.max_set_size = max_set_size

        # +1 because 0 is padding. Real medication ids are shifted by +1.
        self.drug_embeddings = nn.Embedding(
            num_meds + 1,
            embedding_dim,
            padding_idx=0,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # Cardinality matters. A 2-drug set and 15-drug set should not look identical.
        self.count_embedding = nn.Embedding(max_set_size + 1, embedding_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(embedding_dim)

        nn.init.normal_(self.cls_token, std=0.02)

    def _multihot_to_padded_tokens(
        self,
        multihot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts [B, num_meds] multi-hot vectors into padded token ids.

        Returns:
            token_ids: [B, S] LongTensor
            pad_mask: [B, S] BoolTensor, True where padded
            counts:   [B] LongTensor
        """
        device = multihot.device
        selected_rows = []
        counts = []

        for row in multihot.bool():
            idx = torch.nonzero(row, as_tuple=False).flatten()

            if idx.numel() > self.max_set_size:
                # Randomly crop large medication sets during training.
                # Keeps compute bounded and acts as light regularisation.
                perm = torch.randperm(idx.numel(), device=device)
                idx = idx[perm[: self.max_set_size]]

            # Shift by +1 because 0 is padding.
            selected_rows.append(idx + 1)
            counts.append(idx.numel())

        max_len = max([x.numel() for x in selected_rows] + [1])
        max_len = min(max_len, self.max_set_size)

        token_ids = torch.zeros(
            len(selected_rows),
            max_len,
            dtype=torch.long,
            device=device,
        )

        pad_mask = torch.ones(
            len(selected_rows),
            max_len,
            dtype=torch.bool,
            device=device,
        )

        for i, ids in enumerate(selected_rows):
            ids = ids[:max_len]
            n = ids.numel()

            if n > 0:
                token_ids[i, :n] = ids
                pad_mask[i, :n] = False

        counts = torch.tensor(counts, dtype=torch.long, device=device)
        counts = counts.clamp(max=self.max_set_size)

        return token_ids, pad_mask, counts

    def forward(self, multihot: torch.Tensor) -> torch.Tensor:
        token_ids, pad_mask, counts = self._multihot_to_padded_tokens(multihot)

        drug_tokens = self.drug_embeddings(token_ids)

        batch_size = multihot.size(0)

        cls = self.cls_token.expand(batch_size, -1, -1)
        cls = cls + self.count_embedding(counts).unsqueeze(1)

        tokens = torch.cat([cls, drug_tokens], dim=1)

        cls_pad_mask = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=multihot.device,
        )

        full_pad_mask = torch.cat([cls_pad_mask, pad_mask], dim=1)

        encoded = self.encoder(
            tokens,
            src_key_padding_mask=full_pad_mask,
        )

        z_target = encoded[:, 0]
        return self.norm(z_target)


class MedicationSetDecoder(nn.Module):
    """
    Reconstructs the medication multi-hot vector from the target latent.

    This forces the target space to preserve actual medication-set information.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_meds: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_meds),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DrugPredictionHead(nn.Module):
    """
    Simple linear head sitting directly on the context encoder output.
    Trained in the supervised stages; has no dependency on the JEPA predictor.
    """

    def __init__(self, embedding_dim: int, num_meds: int) -> None:
        super().__init__()
        self.net = nn.Linear(embedding_dim, num_meds)

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        return self.net(z_context)  # logits


class VICRegLoss(nn.Module):
    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 0.04,
    ) -> None:
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward_components(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D = z_pred.shape

        sim_loss = F.mse_loss(z_pred, z_target)

        def variance_loss(z: torch.Tensor) -> torch.Tensor:
            return F.relu(1.0 - z.std(dim=0)).mean()

        var_loss = variance_loss(z_pred) + variance_loss(z_target)

        def covariance_loss(z: torch.Tensor) -> torch.Tensor:
            z = z - z.mean(dim=0)
            cov = (z.T @ z) / (B - 1)
            off_diag = cov ** 2
            off_diag.fill_diagonal_(0.0)
            return off_diag.sum() / D

        cov_loss = covariance_loss(z_pred) + covariance_loss(z_target)

        return sim_loss, var_loss, cov_loss

    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        sim_loss, var_loss, cov_loss = self.forward_components(z_pred, z_target)
        return (
            self.sim_weight * sim_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
        )


class DrugJEPA(nn.Module):
    """
    JEPA-style drug recommendation model.

    Architecture
    ------------
    Stage 1 — JEPA pretraining:
        features -> context_encoder -> jepa_predictor -> z_pred
        targets  -> drug_embeddings (mean pooling)    -> z_target
        Loss: VICReg(z_pred, z_target)

        The context encoder learns to produce representations that live
        near the drug embedding manifold.  The jepa_predictor is the
        adaptor that bridges the context space and the drug space.

    Stage 2 — supervised head training:
        features -> context_encoder [frozen] -> prediction_head -> logits

        A fresh linear head is trained on top of the pretrained context
        encoder.  The jepa_predictor is discarded — it was a pretraining
        scaffold, not part of the inference path.

    Stage 3 — fine-tuning:
        features -> context_encoder [unfrozen] -> prediction_head -> logits

    The drug_embeddings table is only used during JEPA pretraining to
    construct meaningful targets.  It plays no role at inference.

    Modules
    -------
    context_encoder  : clinical context -> embedding
    jepa_predictor   : context embedding -> drug-space embedding  (JEPA only)
    drug_embeddings  : per-drug embedding table                   (JEPA only)
    prediction_head  : context embedding -> medication logits     (supervised)

    Freeze API
    ----------
    model.freeze("context_encoder")
    model.freeze("jepa_predictor")
    model.freeze("drug_embeddings")
    model.freeze("prediction_head")
    model.unfreeze(...)
    model.freeze_all() / unfreeze_all()
    model.frozen_modules() -> set[str]
    """

    MODULE_NAMES: tuple[str, ...] = (
        "context_encoder",
        "jepa_predictor",
        "target_encoder",
        "target_decoder",
        "prediction_head",
    )

    def __init__(
        self,
        context_dim: int,
        num_meds: int,
        hidden_dim: int = 512,
        embedding_dim: int = 128,
        use_predictor_for_supervised: bool = False,
        vicreg_sim_weight: float = 25.0,
        vicreg_var_weight: float = 25.0,
        vicreg_cov_weight: float = 0.04,
    ) -> None:
        super().__init__()
        
        self.use_predictor_for_supervised = use_predictor_for_supervised

        self.context_encoder = ContextEncoder(
            input_dim=context_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )

        self.jepa_predictor = JEPAPredictor(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

        self.target_encoder = DrugSetTargetEncoder(
            num_meds=num_meds,
            embedding_dim=embedding_dim,
            num_heads=4,
            num_layers=2,
            max_set_size=64,
        )

        self.target_decoder = MedicationSetDecoder(
            embedding_dim=embedding_dim,
            num_meds=num_meds,
            hidden_dim=hidden_dim,
        )

        self.prediction_head = DrugPredictionHead(
            embedding_dim=embedding_dim,
            num_meds=num_meds,
        )

        self.vicreg = VICRegLoss(
            sim_weight=vicreg_sim_weight,
            var_weight=vicreg_var_weight,
            cov_weight=vicreg_cov_weight,
        )

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    def forward_jepa(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Stage 1: JEPA pretraining.

        The target encoder is expected to have been pretrained to reconstruct
        medication sets. During JEPA, we detach z_target so the context encoder
        must chase a stable target space instead of moving the target space itself.
        """

        targets = targets.float()

        z_context = self.context_encoder(features)
        z_pred = self.jepa_predictor(z_context)

        z_target = self.target_encoder(targets)

        # Important:
        # The JEPA loss should not freely reshape the target space.
        z_target_for_jepa = z_target.detach()

        sim_loss, var_loss, cov_loss = self.vicreg.forward_components(
            z_pred,
            z_target_for_jepa,
        )

        jepa_loss = (
            self.vicreg.sim_weight * sim_loss
            + self.vicreg.var_weight * var_loss
            + self.vicreg.cov_weight * cov_loss
        )

        std = z_pred.std(dim=0)

        return {
            "loss": jepa_loss,
            "jepa_sim_loss": (self.vicreg.sim_weight * sim_loss).detach(),
            "jepa_var_loss": (self.vicreg.var_weight * var_loss).detach(),
            "jepa_cov_loss": (self.vicreg.cov_weight * cov_loss).detach(),

            "z_pred_std": std.mean().detach(),
            "z_pred_std_max": std.max().detach(),
            "z_pred_std_min": std.min().detach(),

            # Return both names to avoid the z_med_std / z_target_std logging mismatch.
            "z_target_std": z_target.std(dim=0).mean().detach(),
            "z_med_std": z_target.std(dim=0).mean().detach(),

            "z_alignment": F.cosine_similarity(
                z_pred,
                z_target_for_jepa,
                dim=-1,
            ).mean().detach(),

            "z_context": z_context,
            "z_pred": z_pred,
            "z_target": z_target.detach(),
        }

    def forward_supervised(
        self,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_context = self.context_encoder(features)

        if self.use_predictor_for_supervised:
            z_for_prediction = self.jepa_predictor(z_context)
        else:
            z_for_prediction = z_context

        logits = self.prediction_head(z_for_prediction)

        return {
            "predictions": logits,
            "z_context": z_context,
            "z_for_prediction": z_for_prediction,
        }

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_supervised(features)

    # ------------------------------------------------------------------
    # Granular freeze API
    # ------------------------------------------------------------------

    def _get_module(self, name: str) -> nn.Module:
        if name not in self.MODULE_NAMES:
            raise ValueError(
                f"Unknown module '{name}'. Choose from: {self.MODULE_NAMES}"
            )
        return getattr(self, name)

    def freeze(self, *names: str) -> None:
        for name in names:
            for param in self._get_module(name).parameters():
                param.requires_grad = False

    def unfreeze(self, *names: str) -> None:
        for name in names:
            for param in self._get_module(name).parameters():
                param.requires_grad = True

    def freeze_all(self) -> None:
        self.freeze(*self.MODULE_NAMES)

    def unfreeze_all(self) -> None:
        self.unfreeze(*self.MODULE_NAMES)

    def frozen_modules(self) -> set[str]:
        result = set()
        for name in self.MODULE_NAMES:
            params = list(getattr(self, name).parameters())
            if params and all(not p.requires_grad for p in params):
                result.add(name)
        return result

    # ------------------------------------------------------------------
    # Convenience / legacy compatibility
    # ------------------------------------------------------------------

    def freeze_jepa_modules(self) -> None:
        """Freeze jepa_predictor and drug_embeddings (JEPA-only modules)."""
        self.freeze("jepa_predictor", "drug_embeddings")

    def unfreeze_jepa_modules(self) -> None:
        self.unfreeze("jepa_predictor", "drug_embeddings")

    def freeze_context_encoder(self) -> None:
        self.freeze("context_encoder")

    def unfreeze_context_encoder(self) -> None:
        self.unfreeze("context_encoder")