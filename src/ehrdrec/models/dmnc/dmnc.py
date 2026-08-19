from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ehrdrec.models.base import TorchEHRDrecModel


# =====================================================================
# DNC memory
# =====================================================================


@dataclass
class _MemoryState:
    memory: torch.Tensor
    usage: torch.Tensor
    precedence: torch.Tensor
    link: torch.Tensor
    write_weighting: torch.Tensor
    read_weightings: torch.Tensor
    read_vectors: torch.Tensor


class _DNCMemory(nn.Module):
    """
    PyTorch implementation of the DNC memory operations used by DMNC.

    The equations follow the TensorFlow ``memory.py`` implementation from
    the original DMNC repository:

        https://github.com/thaihungle/DMNC

    State shapes
    ------------
    memory:
        (batch, memory_words, memory_word_size)

    usage:
        (batch, memory_words)

    precedence:
        (batch, memory_words)

    link:
        (batch, memory_words, memory_words)

    write_weighting:
        (batch, memory_words)

    read_weightings:
        (batch, memory_words, read_heads)

    read_vectors:
        (batch, memory_word_size, read_heads)
    """

    def __init__(
        self,
        memory_words: int,
        memory_word_size: int,
        read_heads: int,
    ) -> None:
        super().__init__()

        self.memory_words = memory_words
        self.memory_word_size = memory_word_size
        self.read_heads = read_heads

        self.register_buffer(
            "_identity",
            torch.eye(
                memory_words,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> _MemoryState:
        return _MemoryState(
            memory=torch.full(
                (
                    batch_size,
                    self.memory_words,
                    self.memory_word_size,
                ),
                1e-6,
                device=device,
                dtype=dtype,
            ),
            usage=torch.zeros(
                (
                    batch_size,
                    self.memory_words,
                ),
                device=device,
                dtype=dtype,
            ),
            precedence=torch.zeros(
                (
                    batch_size,
                    self.memory_words,
                ),
                device=device,
                dtype=dtype,
            ),
            link=torch.zeros(
                (
                    batch_size,
                    self.memory_words,
                    self.memory_words,
                ),
                device=device,
                dtype=dtype,
            ),
            write_weighting=torch.full(
                (
                    batch_size,
                    self.memory_words,
                ),
                1e-6,
                device=device,
                dtype=dtype,
            ),
            read_weightings=torch.full(
                (
                    batch_size,
                    self.memory_words,
                    self.read_heads,
                ),
                1e-6,
                device=device,
                dtype=dtype,
            ),
            read_vectors=torch.full(
                (
                    batch_size,
                    self.memory_word_size,
                    self.read_heads,
                ),
                1e-6,
                device=device,
                dtype=dtype,
            ),
        )

    def zero_read(
        self,
        state: _MemoryState,
    ) -> _MemoryState:
        """
        Match the original ``read_zero`` behaviour while preserving the
        actual memory content and write-state variables.
        """

        return _MemoryState(
            memory=state.memory,
            usage=state.usage,
            precedence=state.precedence,
            link=state.link,
            write_weighting=state.write_weighting,
            read_weightings=torch.full_like(
                state.read_weightings,
                1e-6,
            ),
            read_vectors=torch.full_like(
                state.read_vectors,
                1e-6,
            ),
        )

    @staticmethod
    def _content_weighting(
        memory: torch.Tensor,
        keys: torch.Tensor,
        strengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Content-based lookup.

        Parameters
        ----------
        memory:
            (B, N, W)

        keys:
            (B, W, K)

        strengths:
            (B, K)

        Returns
        -------
        torch.Tensor
            (B, N, K)
        """

        normalised_memory = F.normalize(
            memory,
            p=2,
            dim=2,
            eps=1e-8,
        )

        normalised_keys = F.normalize(
            keys,
            p=2,
            dim=1,
            eps=1e-8,
        )

        similarity = torch.bmm(
            normalised_memory,
            normalised_keys,
        )

        return F.softmax(
            similarity
            * strengths.unsqueeze(1),
            dim=1,
        )

    @staticmethod
    def _update_usage(
        usage: torch.Tensor,
        read_weightings: torch.Tensor,
        write_weighting: torch.Tensor,
        free_gates: torch.Tensor,
    ) -> torch.Tensor:
        retention = torch.prod(
            1.0
            - read_weightings
            * free_gates.unsqueeze(1),
            dim=2,
        )

        return (
            usage
            + write_weighting
            - usage * write_weighting
        ) * retention

    @staticmethod
    def _allocation_weighting(
        usage: torch.Tensor,
    ) -> torch.Tensor:
        sorted_usage, indices = torch.sort(
            usage,
            dim=1,
            descending=False,
        )

        exclusive_cumprod = torch.cat(
            [
                torch.ones(
                    (
                        usage.size(0),
                        1,
                    ),
                    device=usage.device,
                    dtype=usage.dtype,
                ),
                torch.cumprod(
                    sorted_usage,
                    dim=1,
                )[:, :-1],
            ],
            dim=1,
        )

        unordered = (
            1.0 - sorted_usage
        ) * exclusive_cumprod

        allocation = torch.zeros_like(
            usage
        )

        allocation.scatter_(
            1,
            indices,
            unordered,
        )

        return allocation

    @staticmethod
    def _update_write_weighting(
        lookup_weighting: torch.Tensor,
        allocation_weighting: torch.Tensor,
        write_gate: torch.Tensor,
        allocation_gate: torch.Tensor,
    ) -> torch.Tensor:
        lookup_weighting = lookup_weighting.squeeze(
            -1
        )

        return write_gate * (
            allocation_gate * allocation_weighting
            + (
                1.0 - allocation_gate
            ) * lookup_weighting
        )

    @staticmethod
    def _update_memory(
        memory: torch.Tensor,
        write_weighting: torch.Tensor,
        write_vector: torch.Tensor,
        erase_vector: torch.Tensor,
    ) -> torch.Tensor:
        write_weighting = write_weighting.unsqueeze(
            2
        )

        write_vector = write_vector.unsqueeze(
            1
        )

        erase_vector = erase_vector.unsqueeze(
            1
        )

        erased = memory * (
            1.0
            - torch.bmm(
                write_weighting,
                erase_vector,
            )
        )

        written = torch.bmm(
            write_weighting,
            write_vector,
        )

        return erased + written

    @staticmethod
    def _update_precedence(
        precedence: torch.Tensor,
        write_weighting: torch.Tensor,
    ) -> torch.Tensor:
        reset_factor = (
            1.0
            - write_weighting.sum(
                dim=1,
                keepdim=True,
            )
        )

        return (
            reset_factor * precedence
            + write_weighting
        )

    def _update_link(
        self,
        precedence: torch.Tensor,
        link: torch.Tensor,
        write_weighting: torch.Tensor,
    ) -> torch.Tensor:
        wi = write_weighting.unsqueeze(
            2
        )

        wj = write_weighting.unsqueeze(
            1
        )

        reset_factor = (
            1.0 - wi - wj
        )

        updated = (
            reset_factor * link
            + torch.bmm(
                wi,
                precedence.unsqueeze(1),
            )
        )

        identity = self._identity.to(
            device=updated.device,
            dtype=updated.dtype,
        ).unsqueeze(0)

        return updated * (
            1.0 - identity
        )

    def write(
        self,
        state: _MemoryState,
        interface: dict[str, torch.Tensor],
    ) -> _MemoryState:
        lookup_weighting = (
            self._content_weighting(
                state.memory,
                interface["write_key"],
                interface["write_strength"],
            )
        )

        usage = self._update_usage(
            state.usage,
            state.read_weightings,
            state.write_weighting,
            interface["free_gates"],
        )

        allocation = (
            self._allocation_weighting(
                usage
            )
        )

        write_weighting = (
            self._update_write_weighting(
                lookup_weighting,
                allocation,
                interface[
                    "write_gate"
                ],
                interface[
                    "allocation_gate"
                ],
            )
        )

        memory = self._update_memory(
            state.memory,
            write_weighting,
            interface[
                "write_vector"
            ],
            interface[
                "erase_vector"
            ],
        )

        link = self._update_link(
            state.precedence,
            state.link,
            write_weighting,
        )

        precedence = (
            self._update_precedence(
                state.precedence,
                write_weighting,
            )
        )

        return _MemoryState(
            memory=memory,
            usage=usage,
            precedence=precedence,
            link=link,
            write_weighting=write_weighting,
            read_weightings=state.read_weightings,
            read_vectors=state.read_vectors,
        )

    def read(
        self,
        state: _MemoryState,
        interface: dict[str, torch.Tensor],
    ) -> _MemoryState:
        lookup = self._content_weighting(
            state.memory,
            interface["read_keys"],
            interface["read_strengths"],
        )

        forward = torch.bmm(
            state.link,
            state.read_weightings,
        )

        backward = torch.bmm(
            state.link.transpose(
                1,
                2,
            ),
            state.read_weightings,
        )

        read_modes = interface[
            "read_modes"
        ]

        read_weightings = (
            read_modes[
                :,
                0,
                :,
            ].unsqueeze(1)
            * backward
            + read_modes[
                :,
                1,
                :,
            ].unsqueeze(1)
            * lookup
            + read_modes[
                :,
                2,
                :,
            ].unsqueeze(1)
            * forward
        )

        read_vectors = torch.bmm(
            state.memory.transpose(
                1,
                2,
            ),
            read_weightings,
        )

        return _MemoryState(
            memory=state.memory,
            usage=state.usage,
            precedence=state.precedence,
            link=state.link,
            write_weighting=state.write_weighting,
            read_weightings=read_weightings,
            read_vectors=read_vectors,
        )


# =====================================================================
# DNC controller
# =====================================================================


class _DNCController(nn.Module):
    """
    LSTM controller matching the original DMNC recurrent controller.

    The controller maps:

        [input ; previous memory read vectors]
                -> LSTM
                -> pre-output
                -> DNC interface vector

    A decoder controller can address two memories simultaneously.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        memory_word_size: int,
        read_heads: int,
        use_memory: bool,
        memory_count: int = 1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.memory_word_size = memory_word_size
        self.read_heads = read_heads
        self.use_memory = use_memory
        self.memory_count = memory_count

        memory_read_dim = (
            memory_count
            * memory_word_size
            * read_heads
            if use_memory
            else 0
        )

        self.lstm = nn.LSTMCell(
            input_dim
            + memory_read_dim,
            hidden_dim,
        )

        self.output_projection = nn.Linear(
            hidden_dim,
            output_dim,
            bias=False,
        )

        self.interface_dim = (
            memory_word_size
            * read_heads
            + 3 * memory_word_size
            + 5 * read_heads
            + 3
        )

        self.interface_projection = nn.Linear(
            hidden_dim,
            self.interface_dim
            * memory_count,
            bias=False,
        )

        self.memory_output_projection = (
            nn.Linear(
                memory_read_dim,
                output_dim,
                bias=False,
            )
            if use_memory
            else None
        )

        self._initialise_weights()

    def _initialise_weights(
        self,
    ) -> None:
        nn.init.normal_(
            self.output_projection.weight,
            std=0.1,
        )

        nn.init.normal_(
            self.interface_projection.weight,
            std=0.1,
        )

        if (
            self.memory_output_projection
            is not None
        ):
            nn.init.normal_(
                self.memory_output_projection.weight,
                std=0.1,
            )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        h = torch.zeros(
            (
                batch_size,
                self.hidden_dim,
            ),
            device=device,
            dtype=dtype,
        )

        c = torch.zeros_like(
            h
        )

        return h, c

    def parse_interface(
        self,
        interface: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        word_size = (
            self.memory_word_size
        )

        read_heads = (
            self.read_heads
        )

        cursor = 0

        read_keys_size = (
            word_size * read_heads
        )

        read_keys = interface[
            :,
            cursor:
            cursor + read_keys_size,
        ].reshape(
            -1,
            word_size,
            read_heads,
        )

        cursor += read_keys_size

        read_strengths = interface[
            :,
            cursor:
            cursor + read_heads,
        ]

        cursor += read_heads

        write_key = interface[
            :,
            cursor:
            cursor + word_size,
        ].reshape(
            -1,
            word_size,
            1,
        )

        cursor += word_size

        write_strength = interface[
            :,
            cursor:
            cursor + 1,
        ]

        cursor += 1

        erase_vector = interface[
            :,
            cursor:
            cursor + word_size,
        ]

        cursor += word_size

        write_vector = interface[
            :,
            cursor:
            cursor + word_size,
        ]

        cursor += word_size

        free_gates = interface[
            :,
            cursor:
            cursor + read_heads,
        ]

        cursor += read_heads

        allocation_gate = interface[
            :,
            cursor:
            cursor + 1,
        ]

        cursor += 1

        write_gate = interface[
            :,
            cursor:
            cursor + 1,
        ]

        cursor += 1

        read_modes = interface[
            :,
            cursor:,
        ].reshape(
            -1,
            3,
            read_heads,
        )

        return {
            "read_keys": read_keys,
            "read_strengths": (
                1.0
                + F.softplus(
                    read_strengths
                )
            ),
            "write_key": write_key,
            "write_strength": (
                1.0
                + F.softplus(
                    write_strength
                )
            ),
            "erase_vector": (
                torch.sigmoid(
                    erase_vector
                )
            ),
            "write_vector": (
                write_vector
            ),
            "free_gates": (
                torch.sigmoid(
                    free_gates
                )
            ),
            "allocation_gate": (
                torch.sigmoid(
                    allocation_gate
                )
            ),
            "write_gate": (
                torch.sigmoid(
                    write_gate
                )
            ),
            "read_modes": F.softmax(
                read_modes,
                dim=1,
            ),
        }

    def process(
        self,
        x: torch.Tensor,
        read_vectors: list[
            torch.Tensor
        ],
        state: tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[
        torch.Tensor,
        list[
            dict[str, torch.Tensor]
        ],
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        if self.use_memory:
            flat_reads = [
                tensor.reshape(
                    tensor.size(0),
                    -1,
                )
                for tensor
                in read_vectors
            ]

            controller_input = torch.cat(
                [
                    x,
                    *flat_reads,
                ],
                dim=-1,
            )
        else:
            controller_input = x

        h, c = self.lstm(
            controller_input,
            state,
        )

        pre_output = (
            self.output_projection(
                h
            )
        )

        raw_interface = (
            self.interface_projection(
                h
            )
        )

        interface_chunks = (
            raw_interface.chunk(
                self.memory_count,
                dim=-1,
            )
        )

        interfaces = [
            self.parse_interface(
                chunk
            )
            for chunk
            in interface_chunks
        ]

        return (
            pre_output,
            interfaces,
            (h, c),
        )

    def final_output(
        self,
        pre_output: torch.Tensor,
        read_vectors: list[
            torch.Tensor
        ],
    ) -> torch.Tensor:
        if not self.use_memory:
            return pre_output

        assert (
            self.memory_output_projection
            is not None
        )

        flattened = torch.cat(
            [
                tensor.reshape(
                    tensor.size(0),
                    -1,
                )
                for tensor
                in read_vectors
            ],
            dim=-1,
        )

        return (
            pre_output
            + self.memory_output_projection(
                flattened
            )
        )


# =====================================================================
# EHRDRec DMNC
# =====================================================================


class DMNC(TorchEHRDrecModel):
    """
    PyTorch / EHRDRec implementation of the Dual Memory Neural Computer.

    Source implementation
    ---------------------
    Hung Le, Truyen Tran, and Svetha Venkatesh.
    "Dual Memory Neural Computer for Asynchronous Two-view Sequential
    Learning." KDD 2018.

    Original TensorFlow implementation:
        https://github.com/thaihungle/DMNC

    This class ports the medication-recommendation configuration used by the
    original repository rather than retaining TensorFlow graph/session APIs.

    EHRDRec patient input
    ---------------------
    ``x`` is an ordered patient history:

        [
            {
                "diagnoses": list[int],
                "procedures": list[int],
            },
            ...
        ]

    The final visit in the history is the visit for which medication logits
    are returned.

    Important
    ---------
    DMNC processes diagnosis and procedure codes *sequentially*, one code per
    DNC timestep. Therefore the order of the codes within each visit is part
    of the model input.

    The two sequences are right-aligned exactly as in the original MIMIC
    preparation code, with the shorter sequence entering the computation
    later so both views reach their end-of-view token at the same timestep.

    Medication target
    -----------------
    For the medication-recommendation experiment, the source implementation
    uses ``multi=True`` and collapses all medications for the admission into
    one multi-hot target. The training objective is therefore sigmoid binary
    cross entropy, not autoregressive medication generation.

        batch["Y"] -> Tensor[medications_vocab_size]

    Historical medications are NOT a patient input to DMNC.

    External resources
    ------------------
    None.

    Notes on scope
    --------------
    The original ``Dual_DNC`` class is a general-purpose implementation with
    several modes that are not used by its multi-label medication experiment
    (autoregressive decoder mode, teacher forcing, sequence-level softmax
    objectives, etc.). They are intentionally not exposed here.

    This implementation corresponds to the late-fusion DMNC configuration
    (separate DNC memories).
    """

    def __init__(
        self,
        diagnoses_vocab_size: int,
        procedures_vocab_size: int,
        medications_vocab_size: int,
        embedding_dim: int = 64,
        memory_words: int = 16,
        memory_word_size: int = 64,
        memory_read_heads: int = 1,
        hidden_controller_dim: int = 64,
        use_memory: bool = True,
        share_memory: bool = False,
        attend_dim: int = 0,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        gradient_clip_value: float = 10.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if share_memory:
            raise NotImplementedError(
                "The TensorFlow source does not define "
                "unambiguous shared-state semantics for DMNC_e. "
                "This EHRDRec port currently implements the "
                "late-fusion DMNC configuration "
                "(share_memory=False)."
            )

        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.diagnoses_vocab_size = (
            diagnoses_vocab_size
        )

        self.procedures_vocab_size = (
            procedures_vocab_size
        )

        self.medications_vocab_size = (
            medications_vocab_size
        )

        self.embedding_dim = embedding_dim
        self.memory_words = memory_words
        self.memory_word_size = (
            memory_word_size
        )
        self.memory_read_heads = (
            memory_read_heads
        )
        self.hidden_controller_dim = (
            hidden_controller_dim
        )
        self.use_memory = use_memory
        self.attend_dim = attend_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.gradient_clip_value = (
            gradient_clip_value
        )

        # ------------------------------------------------------------
        # Special input tokens
        # ------------------------------------------------------------
        #
        # The original code reserves:
        #     0 = padding / blank
        #     1 = end-of-view
        #
        # EHRDRec vocabularies need not reserve these IDs, so append explicit
        # special tokens after the real vocabulary instead.

        self.DIAG_PAD_TOKEN = (
            diagnoses_vocab_size
        )
        self.DIAG_END_TOKEN = (
            diagnoses_vocab_size + 1
        )

        self.PROC_PAD_TOKEN = (
            procedures_vocab_size
        )
        self.PROC_END_TOKEN = (
            procedures_vocab_size + 1
        )

        self.diagnosis_embedding = (
            nn.Embedding(
                diagnoses_vocab_size + 2,
                embedding_dim,
            )
        )

        self.procedure_embedding = (
            nn.Embedding(
                procedures_vocab_size + 2,
                embedding_dim,
            )
        )

        # The original implementation uses dense one-hot x embedding-matrix
        # multiplication. An embedding lookup is algebraically equivalent.

        self.memory1 = _DNCMemory(
            memory_words=memory_words,
            memory_word_size=(
                memory_word_size
            ),
            read_heads=(
                memory_read_heads
            ),
        )

        self.memory2 = _DNCMemory(
            memory_words=memory_words,
            memory_word_size=(
                memory_word_size
            ),
            read_heads=(
                memory_read_heads
            ),
        )

        self.controller1 = _DNCController(
            input_dim=embedding_dim,
            output_dim=(
                medications_vocab_size
            ),
            hidden_dim=(
                hidden_controller_dim
            ),
            memory_word_size=(
                memory_word_size
            ),
            read_heads=(
                memory_read_heads
            ),
            use_memory=use_memory,
            memory_count=1,
        )

        self.controller2 = _DNCController(
            input_dim=embedding_dim,
            output_dim=(
                medications_vocab_size
            ),
            hidden_dim=(
                hidden_controller_dim
            ),
            memory_word_size=(
                memory_word_size
            ),
            read_heads=(
                memory_read_heads
            ),
            use_memory=use_memory,
            memory_count=1,
        )

        decoder_input_dim = (
            embedding_dim
        )

        if attend_dim > 0:
            self.attention_w1 = nn.Linear(
                hidden_controller_dim,
                attend_dim,
                bias=False,
            )
            self.attention_u1 = nn.Linear(
                hidden_controller_dim,
                attend_dim,
                bias=False,
            )
            self.attention_v1 = nn.Parameter(
                torch.empty(
                    attend_dim
                )
            )

            self.attention_w2 = nn.Linear(
                hidden_controller_dim,
                attend_dim,
                bias=False,
            )
            self.attention_u2 = nn.Linear(
                hidden_controller_dim,
                attend_dim,
                bias=False,
            )
            self.attention_v2 = nn.Parameter(
                torch.empty(
                    attend_dim
                )
            )

            decoder_input_dim += (
                2
                * hidden_controller_dim
            )

            nn.init.uniform_(
                self.attention_v1,
                -1.0,
                1.0,
            )
            nn.init.uniform_(
                self.attention_v2,
                -1.0,
                1.0,
            )

        self.decoder_controller = (
            _DNCController(
                input_dim=(
                    decoder_input_dim
                ),
                output_dim=(
                    medications_vocab_size
                ),
                hidden_dim=(
                    2
                    * hidden_controller_dim
                ),
                memory_word_size=(
                    memory_word_size
                ),
                read_heads=(
                    memory_read_heads
                ),
                use_memory=(
                    use_memory
                ),
                memory_count=2,
            )
        )

        self._initialise_embeddings()

        self.to(
            self.device
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
        )

    # ================================================================
    # Initialisation
    # ================================================================

    def _initialise_embeddings(
        self,
    ) -> None:
        # The source initialises embedding matrices with uniform(-1, 1).
        nn.init.uniform_(
            self.diagnosis_embedding.weight,
            -1.0,
            1.0,
        )

        nn.init.uniform_(
            self.procedure_embedding.weight,
            -1.0,
            1.0,
        )

    # ================================================================
    # Input validation
    # ================================================================

    @staticmethod
    def _normalise_history(
        patient_history: Any,
    ) -> list[
        dict[str, list[int]]
    ]:
        if not isinstance(
            patient_history,
            (list, tuple),
        ):
            raise TypeError(
                "DMNC input must be an ordered list "
                "of visit dictionaries."
            )

        history = list(
            patient_history
        )

        if not history:
            raise ValueError(
                "DMNC requires at least one visit."
            )

        for visit in history:
            if not isinstance(
                visit,
                dict,
            ):
                raise TypeError(
                    "Each DMNC visit must be a dictionary."
                )

            if "diagnoses" not in visit:
                raise KeyError(
                    "Each DMNC visit requires "
                    "'diagnoses'."
                )

            if "procedures" not in visit:
                raise KeyError(
                    "Each DMNC visit requires "
                    "'procedures'."
                )

            if len(
                visit["diagnoses"]
            ) == 0:
                raise ValueError(
                    "DMNC requires at least one "
                    "diagnosis code per visit."
                )

            # The original preprocessing can tolerate an empty second view
            # by substituting a blank token. Preserve that capability here.

        return history

    # ================================================================
    # Attention
    # ================================================================

    def _attention_context(
        self,
        encoder_hiddens: list[
            torch.Tensor
        ],
        state_h: torch.Tensor,
        view: int,
    ) -> torch.Tensor:
        if not encoder_hiddens:
            return torch.zeros_like(
                state_h
            )

        encoded = torch.stack(
            encoder_hiddens,
            dim=1,
        )

        if view == 1:
            projected_history = (
                self.attention_u1(
                    encoded
                )
            )
            projected_state = (
                self.attention_w1(
                    state_h
                )
            ).unsqueeze(1)
            vector = self.attention_v1
        else:
            projected_history = (
                self.attention_u2(
                    encoded
                )
            )
            projected_state = (
                self.attention_w2(
                    state_h
                )
            ).unsqueeze(1)
            vector = self.attention_v2

        energy = torch.tanh(
            projected_history
            + projected_state
        )

        scores = torch.matmul(
            energy,
            vector,
        )

        weights = F.softmax(
            scores,
            dim=1,
        )

        return torch.sum(
            encoded
            * weights.unsqueeze(-1),
            dim=1,
        )

    # ================================================================
    # One encoder-memory step
    # ================================================================

    def _encoder_step(
        self,
        embedding: torch.Tensor,
        controller: _DNCController,
        memory_module: _DNCMemory,
        controller_state: tuple[
            torch.Tensor,
            torch.Tensor,
        ],
        memory_state: _MemoryState,
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
        _MemoryState,
        torch.Tensor,
    ]:
        (
            _pre_output,
            interfaces,
            controller_state,
        ) = controller.process(
            embedding,
            [
                memory_state.read_vectors
            ],
            controller_state,
        )

        if self.use_memory:
            memory_state = (
                memory_module.write(
                    memory_state,
                    interfaces[0],
                )
            )

            memory_state = (
                memory_module.read(
                    memory_state,
                    interfaces[0],
                )
            )

        return (
            controller_state,
            memory_state,
            controller_state[0],
        )

    # ================================================================
    # One admission
    # ================================================================

    def _process_visit(
        self,
        visit: dict[
            str,
            list[int]
        ],
        controller_state1: tuple[
            torch.Tensor,
            torch.Tensor,
        ],
        controller_state2: tuple[
            torch.Tensor,
            torch.Tensor,
        ],
        memory_state1: _MemoryState,
        memory_state2: _MemoryState,
    ) -> tuple[
        torch.Tensor,
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
        _MemoryState,
        _MemoryState,
    ]:
        diagnosis_codes = list(
            visit["diagnoses"]
        )

        procedure_codes = list(
            visit["procedures"]
        )

        if not procedure_codes:
            # Equivalent to the source code's blank-token fallback.
            procedure_codes = [
                self.PROC_PAD_TOKEN
            ]

        max_length = max(
            len(diagnosis_codes),
            len(procedure_codes),
        )

        diagnosis_start = (
            max_length
            - len(diagnosis_codes)
        )

        procedure_start = (
            max_length
            - len(procedure_codes)
        )

        diagnosis_sequence = (
            [
                self.DIAG_PAD_TOKEN
            ]
            * diagnosis_start
            + diagnosis_codes
            + [
                self.DIAG_END_TOKEN
            ]
        )

        procedure_sequence = (
            [
                self.PROC_PAD_TOKEN
            ]
            * procedure_start
            + procedure_codes
            + [
                self.PROC_END_TOKEN
            ]
        )

        hidden_history1: list[
            torch.Tensor
        ] = []

        hidden_history2: list[
            torch.Tensor
        ] = []

        # ------------------------------------------------------------
        # Encode the two asynchronous views.
        # ------------------------------------------------------------

        for timestep in range(
            max_length + 1
        ):
            # View 1
            if timestep < diagnosis_start:
                controller_state1 = (
                    self.controller1.initial_state(
                        batch_size=1,
                        device=self.device,
                    )
                )

                memory_state1 = (
                    self.memory1.zero_read(
                        memory_state1
                    )
                )

            else:
                token = torch.tensor(
                    [
                        diagnosis_sequence[
                            timestep
                        ]
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                embedding = (
                    self.diagnosis_embedding(
                        token
                    )
                )

                (
                    controller_state1,
                    memory_state1,
                    hidden1,
                ) = self._encoder_step(
                    embedding,
                    self.controller1,
                    self.memory1,
                    controller_state1,
                    memory_state1,
                )

                hidden_history1.append(
                    hidden1
                )

            # View 2
            if timestep < procedure_start:
                controller_state2 = (
                    self.controller2.initial_state(
                        batch_size=1,
                        device=self.device,
                    )
                )

                memory_state2 = (
                    self.memory2.zero_read(
                        memory_state2
                    )
                )

            else:
                token = torch.tensor(
                    [
                        procedure_sequence[
                            timestep
                        ]
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                embedding = (
                    self.procedure_embedding(
                        token
                    )
                )

                (
                    controller_state2,
                    memory_state2,
                    hidden2,
                ) = self._encoder_step(
                    embedding,
                    self.controller2,
                    self.memory2,
                    controller_state2,
                    memory_state2,
                )

                hidden_history2.append(
                    hidden2
                )

        # ------------------------------------------------------------
        # One medication multi-label decoder step.
        #
        # The source's multi=True preparation leaves a blank input token at
        # decoder_point. Use the appended diagnosis PAD embedding as the
        # corresponding decoder input.
        # ------------------------------------------------------------

        decoder_token = torch.tensor(
            [
                self.DIAG_PAD_TOKEN
            ],
            dtype=torch.long,
            device=self.device,
        )

        decoder_input = (
            self.diagnosis_embedding(
                decoder_token
            )
        )

        if self.attend_dim > 0:
            context1 = (
                self._attention_context(
                    hidden_history1,
                    controller_state1[0],
                    view=1,
                )
            )

            context2 = (
                self._attention_context(
                    hidden_history2,
                    controller_state2[0],
                    view=2,
                )
            )

            decoder_input = torch.cat(
                [
                    decoder_input,
                    context1,
                    context2,
                ],
                dim=-1,
            )

        combined_h = torch.cat(
            [
                controller_state1[0],
                controller_state2[0],
            ],
            dim=-1,
        )

        combined_c = torch.cat(
            [
                controller_state1[1],
                controller_state2[1],
            ],
            dim=-1,
        )

        (
            pre_output,
            decoder_interfaces,
            decoder_state,
        ) = self.decoder_controller.process(
            decoder_input,
            [
                memory_state1.read_vectors,
                memory_state2.read_vectors,
            ],
            (
                combined_h,
                combined_c,
            ),
        )

        # The medication experiment uses write_protect=True during decoding,
        # so the decoder reads but does not write either external memory.
        if self.use_memory:
            memory_state1 = self.memory1.read(
                memory_state1,
                decoder_interfaces[0],
            )

            memory_state2 = self.memory2.read(
                memory_state2,
                decoder_interfaces[1],
            )

        logits = (
            self.decoder_controller.final_output(
                pre_output,
                [
                    memory_state1.read_vectors,
                    memory_state2.read_vectors,
                ],
            )
        )

        # The source splits the 2H decoder state back into the two H encoder
        # controller states and persists those into the next admission.
        next_h1, next_h2 = (
            decoder_state[0].chunk(
                2,
                dim=-1,
            )
        )

        next_c1, next_c2 = (
            decoder_state[1].chunk(
                2,
                dim=-1,
            )
        )

        return (
            logits,
            (
                next_h1,
                next_c1,
            ),
            (
                next_h2,
                next_c2,
            ),
            memory_state1,
            memory_state2,
        )

    # ================================================================
    # Forward
    # ================================================================

    def forward(
        self,
        patient_history: list[
            dict[str, list[int]]
        ],
    ) -> torch.Tensor:
        """
        Return medication logits for the final visit in ``patient_history``.

        The diagnosis/procedure DNC states and memories are propagated through
        every visit in the history, reproducing the original model's persistent
        per-patient memory behaviour without retaining state across independent
        EHRDRec samples.
        """

        history = self._normalise_history(
            patient_history
        )

        controller_state1 = (
            self.controller1.initial_state(
                batch_size=1,
                device=self.device,
            )
        )

        controller_state2 = (
            self.controller2.initial_state(
                batch_size=1,
                device=self.device,
            )
        )

        memory_state1 = (
            self.memory1.initial_state(
                batch_size=1,
                device=self.device,
            )
        )

        memory_state2 = (
            self.memory2.initial_state(
                batch_size=1,
                device=self.device,
            )
        )

        logits: torch.Tensor | None = (
            None
        )

        for visit in history:
            (
                logits,
                controller_state1,
                controller_state2,
                memory_state1,
                memory_state2,
            ) = self._process_visit(
                visit,
                controller_state1,
                controller_state2,
                memory_state1,
                memory_state2,
            )

        assert logits is not None

        return logits

    # ================================================================
    # Prediction
    # ================================================================

    def predict(
        self,
        x: list[
            dict[str, list[int]]
        ],
    ) -> torch.Tensor:
        """
        Return raw medication logits.

        Sigmoid conversion and thresholding belong to the common EHRDRec
        evaluation layer.
        """

        self.eval()

        with torch.no_grad():
            return self.forward(
                x
            )

    # ================================================================
    # Loss
    # ================================================================

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Multi-label medication loss from the original EMR experiment.
        """

        del kwargs

        target = torch.as_tensor(
            target,
            dtype=torch.float32,
            device=self.device,
        )

        if target.ndim == 1:
            target = target.unsqueeze(
                0
            )

        if pred.shape != target.shape:
            raise ValueError(
                "DMNC target shape must match "
                "the medication logits: "
                f"pred={tuple(pred.shape)}, "
                f"target={tuple(target.shape)}."
            )

        return F.binary_cross_entropy_with_logits(
            pred,
            target,
        )

    # ================================================================
    # EHRDRec training
    # ================================================================

    @staticmethod
    def _normalise_batch_histories(
        x: Any,
    ) -> list[
        list[
            dict[str, list[int]]
        ]
    ]:
        # One history:
        #   [visit, visit, ...]
        if (
            isinstance(
                x,
                (list, tuple),
            )
            and len(x) > 0
            and isinstance(
                x[0],
                dict,
            )
        ):
            return [
                list(x)
            ]

        # Batch of histories:
        #   [[visit, ...], [visit, ...], ...]
        if isinstance(
            x,
            (list, tuple),
        ):
            return [
                list(history)
                for history in x
            ]

        raise TypeError(
            "DMNC batch['x'] must be a patient history "
            "or a list of patient histories."
        )

    def fit(
        self,
        train_data: DataLoader,
        validation_data: DataLoader,
        resources: dict[
            str,
            Any
        ] | None = None,
    ) -> None:
        """
        Train DMNC through the EHRDRec interface.

        Expected batch structure
        ------------------------
        batch["x"]:
            one patient history or a batch of patient histories

        batch["Y"]:
            one multi-hot current-visit medication target per history

        The source code trains with batch size 1. To preserve those update
        semantics, this method performs one optimizer step per history even
        when the EHRDRec DataLoader groups examples together.
        """

        del resources

        for _epoch in range(
            self.epochs
        ):
            self.train()

            for batch in train_data:
                histories = (
                    self._normalise_batch_histories(
                        batch["x"]
                    )
                )

                targets = torch.as_tensor(
                    batch["Y"],
                    dtype=torch.float32,
                    device=self.device,
                )

                if targets.ndim == 1:
                    targets = targets.unsqueeze(
                        0
                    )

                if len(
                    histories
                ) != targets.size(0):
                    raise ValueError(
                        "DMNC received a different number "
                        "of histories and targets."
                    )

                for history, target in zip(
                    histories,
                    targets,
                ):
                    logits = self.forward(
                        history
                    )

                    loss = self.loss(
                        logits,
                        target.unsqueeze(0),
                    )

                    self.optimizer.zero_grad()

                    loss.backward()

                    # Original TensorFlow code clips each gradient element
                    # into [-10, 10].
                    nn.utils.clip_grad_value_(
                        self.parameters(),
                        self.gradient_clip_value,
                    )

                    self.optimizer.step()

            # --------------------------------------------------------
            # Validation pass.
            #
            # EHRDRec should eventually own metric calculation, model
            # selection and early stopping so those choices are common
            # across all benchmark models.
            # --------------------------------------------------------

            self.eval()

            with torch.no_grad():
                for batch in validation_data:
                    histories = (
                        self._normalise_batch_histories(
                            batch["x"]
                        )
                    )

                    targets = torch.as_tensor(
                        batch["Y"],
                        dtype=torch.float32,
                        device=self.device,
                    )

                    if targets.ndim == 1:
                        targets = (
                            targets.unsqueeze(0)
                        )

                    for history, target in zip(
                        histories,
                        targets,
                    ):
                        logits = self.forward(
                            history
                        )

                        _ = self.loss(
                            logits,
                            target.unsqueeze(0),
                        )

    # ================================================================
    # Saving
    # ================================================================

    def save(
        self,
        path: str | Path,
    ) -> None:
        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.state_dict(),
            path,
        )
