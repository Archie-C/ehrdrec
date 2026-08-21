from ehrdrec.utils import ReservedId, Vocab


def test_vocab_encodes_and_decodes_lists_with_unknown_fallbacks() -> None:
    vocab = Vocab(
        token_to_id={"UNK": 0, "PAD": 1, "known": 4},
        id_to_token={0: "UNK", 1: "PAD", 4: "known"},
    )

    assert vocab.encode_list(["known", "missing"]) == [
        4,
        int(ReservedId.UNK),
    ]
    assert vocab.decode_list([4, 99]) == ["known", "UNK"]
    assert vocab.vocab_size == 5
