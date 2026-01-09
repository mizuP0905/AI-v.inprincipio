from typing import List, Sequence

import torch


class BaseTokenizer:
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int

    def encode(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int = None,
    ) -> torch.LongTensor:
        raise NotImplementedError

    def decode(self, ids: torch.LongTensor) -> List[str]:
        raise NotImplementedError


class SimpleTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab: Sequence[str],
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        unk_id: int = 3,
    ) -> None:
        self.vocab = list(vocab)
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(self.vocab)}
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    def encode(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int = None,
    ) -> torch.LongTensor:
        encoded = []
        for text in texts:
            tokens = text.strip().split()
            ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]
            if add_bos:
                ids = [self.bos_id] + ids
            if add_eos:
                ids = ids + [self.eos_id]
            if max_length is not None:
                ids = ids[:max_length]
            encoded.append(ids)

        if max_length is None:
            max_length = max(len(ids) for ids in encoded) if encoded else 0

        padded = []
        for ids in encoded:
            if len(ids) < max_length:
                ids = ids + [self.pad_id] * (max_length - len(ids))
            padded.append(ids)

        return torch.tensor(padded, dtype=torch.long)

    def decode(self, ids: torch.LongTensor) -> List[str]:
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        texts = []
        for row in ids.tolist():
            tokens = []
            for idx in row:
                if idx == self.pad_id:
                    continue
                tokens.append(self.id_to_token.get(idx, "<unk>"))
            texts.append(" ".join(tokens))
        return texts
