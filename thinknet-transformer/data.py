import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Simple dataset that produces (x, y) sequence pairs from a 1-D token tensor.

    Expects `tokens` to be a 1-D tensor or a list/array of token ids. Returns
    slices of length `seq_len` for `x` and the next-token targets for `y`.
    """

    def __init__(self, tokens, seq_len: int):
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)

        self.tokens = tokens.long()
        self.seq_len = int(seq_len)

    def __len__(self):
        # number of available starting positions for full-length examples
        return max(1, self.tokens.numel() - self.seq_len)

    def __getitem__(self, idx):
        start = int(idx)
        end = start + self.seq_len

        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]

        return x, y