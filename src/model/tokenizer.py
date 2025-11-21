class CharTokenizer:
    def __init__(self):
        # Digits 0-9, operators +, -, *, =, (, ), and space/padding
        self.chars = "0123456789+-*=() "
        self.vocab_size = len(self.chars) + 1 # +1 for unknown/pad
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.pad_token_id = len(self.chars) # Use last index as pad/unknown if needed

    def encode(self, s):
        return [self.stoi.get(c, self.pad_token_id) for c in s]

    def decode(self, indices):
        return "".join([self.itos.get(i, "") for i in indices])
