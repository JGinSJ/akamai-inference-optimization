"""
Character-level tokenizer.

Vocabulary: 95 printable ASCII characters (chr(32)–chr(126))
plus three special tokens: <PAD>=0, <BOS>=1, <EOS>=2.
Printable chars are mapped to ids 3–97, giving VOCAB_SIZE=98.

No external dependencies.
"""

import string
from typing import List

# Special token ids
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

# All printable ASCII characters, sorted for a stable mapping
_PRINTABLE = string.printable[:95]  # chr(32)..chr(126), len=95
_CHAR_TO_ID = {ch: i + 3 for i, ch in enumerate(_PRINTABLE)}
_ID_TO_CHAR = {i + 3: ch for i, ch in enumerate(_PRINTABLE)}

VOCAB_SIZE = 3 + len(_PRINTABLE)  # 98


class CharTokenizer:
    """
    Encodes a string as a list of integer token ids.
    Unknown characters are silently dropped.
    """

    vocab_size: int = VOCAB_SIZE
    pad_id: int = PAD_ID
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(BOS_ID)
        for ch in text:
            token_id = _CHAR_TO_ID.get(ch)
            if token_id is not None:
                ids.append(token_id)
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: List[int]) -> str:
        chars = []
        for token_id in ids:
            if token_id in _ID_TO_CHAR:
                chars.append(_ID_TO_CHAR[token_id])
            # Special tokens (PAD, BOS, EOS) produce no output character
        return "".join(chars)
