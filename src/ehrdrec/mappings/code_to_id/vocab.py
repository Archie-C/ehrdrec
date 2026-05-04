from dataclasses import dataclass
import polars as pl

from ehrdrec.utils import ReservedId


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @classmethod
    def from_lazyframe(cls, lf: pl.LazyFrame, col: str) -> "Vocab":
        vocab_df = (
            lf
            .select(pl.col(col).explode().alias("token"))
            .drop_nulls()
            .unique()
            .sort("token")
            .with_row_index("id", offset=2)
            .collect()
        )

        token_to_id = dict(zip(vocab_df["token"], vocab_df["id"]))
        token_to_id["UNK"] = int(ReservedId.UNK)
        token_to_id["PAD"] = int(ReservedId.PAD)

        id_to_token = {v: k for k, v in token_to_id.items()}

        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def encode_expr(self, col: str, out_col: str) -> pl.Expr:
        def encode_tokens(tokens) -> list[int]:
            if tokens is None:
                return [int(ReservedId.UNK)]

            tokens = tokens.to_list() if hasattr(tokens, "to_list") else list(tokens)

            return [
                self.token_to_id.get(str(token), int(ReservedId.UNK))
                for token in tokens
                if token is not None
            ] or [int(ReservedId.UNK)]

        return (
            pl.col(col)
            .map_elements(
                encode_tokens,
                return_dtype=pl.List(pl.Int64),
            )
            .alias(out_col)
        )

    def decode_expr(self, col: str, out_col: str) -> pl.Expr:
        def decode_ids(ids) -> list[str]:
            if ids is None:
                return ["UNK"]

            ids = ids.to_list() if hasattr(ids, "to_list") else list(ids)

            return [
                self.id_to_token.get(int(i), "UNK")
                for i in ids
                if i is not None
            ] or ["UNK"]

        return (
            pl.col(col)
            .map_elements(
                decode_ids,
                return_dtype=pl.List(pl.Utf8),
            )
            .alias(out_col)
        )

    def encode_list(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(x, int(ReservedId.UNK)) for x in tokens]

    def decode_list(self, ids: list[int]) -> list[str]:
        return [self.id_to_token.get(int(x), "UNK") for x in ids]
    
    @property
    def vocab_size(self) -> int:
        return max(self.id_to_token) + 1
    
    def to_multihot_expr(
        self,
        col: str,
        out_col: str,
        include_reserved: bool = True,
    ) -> pl.Expr:
        start_idx = 0 if include_reserved else 2
        size = self.vocab_size - start_idx

        def to_multihot(ids) -> list[int]:
            vec = [0] * size

            if ids is None:
                return vec

            ids = ids.to_list() if hasattr(ids, "to_list") else list(ids)

            for i in ids:
                if i is None:
                    continue

                j = int(i) - start_idx

                if 0 <= j < size:
                    vec[j] = 1

            return vec

        return (
            pl.col(col)
            .map_elements(
                to_multihot,
                return_dtype=pl.List(pl.Int8),
            )
            .alias(out_col)
        )