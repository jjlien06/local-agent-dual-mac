"""Experimental CPU attention witness only; not installed in any runtime."""


class BoundRope:
    def __init__(self, rotary):
        self.rotary = rotary

    def bind(self, positions, *, physical_offset, tokens):
        self.positions = positions
        self.physical_offset = physical_offset
        self.tokens = tokens

    def __call__(self, x, offset=0):
        if offset != self.physical_offset:
            raise ValueError('physical cache offset differs from bound chunk')
        if x.shape[2] != self.tokens:
            raise ValueError('token length differs from bound chunk')
        return self.rotary.apply_rotary(x, x, self.positions)[0]
