from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ehrdrec.utils import ReservedId

@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @property
    def vocab_size(self) -> int:
        """Return the number of IDs in the vocabulary."""

        return max(self.id_to_token, default=-1) + 1

    def encode_list(self, tokens: list[str]) -> list[int]:
        """Encode tokens, mapping unknown values to the reserved UNK ID."""

        unknown_id = int(ReservedId.UNK)
        return [
            self.token_to_id.get(str(token), unknown_id)
            for token in tokens
        ]

    def decode_list(self, ids: list[int]) -> list[str]:
        """Decode IDs, mapping unknown values to the UNK token."""

        return [
            self.id_to_token.get(int(token_id), "UNK")
            for token_id in ids
        ]

    @classmethod
    def from_tokens(
        cls,
        lf: pl.LazyFrame,
        col: str = "token",
    ) -> "Vocab":
        reserved = {
            "PAD": int(ReservedId.PAD),
            "UNK": int(ReservedId.UNK),
            "SOS": int(ReservedId.SOS),
            "EOS": int(ReservedId.EOS),
        }

        first_token_id = max(reserved.values()) + 1

        vocab_df = (
            lf
            .select(pl.col(col).cast(pl.Utf8).alias("token"))
            .drop_nulls()
            .unique()
            .sort("token")
            .with_row_index("id", offset=first_token_id)
            .collect()
        )

        token_to_id = {
            str(token): int(idx)
            for token, idx in zip(vocab_df["token"], vocab_df["id"])
        }

        token_to_id.update(reserved)

        id_to_token = {
            idx: token
            for token, idx in token_to_id.items()
        }

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
        )