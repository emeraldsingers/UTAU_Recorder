from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float32)
        self.index = 0
        self.full = False

    def push(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32).flatten()
        n = len(data)
        if n >= self.size:
            self.buffer[:] = data[-self.size:]
            self.index = 0
            self.full = True
            return

        end = self.index + n
        if end < self.size:
            self.buffer[self.index:end] = data
        else:
            first = self.size - self.index
            self.buffer[self.index:] = data[:first]
            self.buffer[:end % self.size] = data[first:]
        self.index = end % self.size
        if self.full:
            return
        if end >= self.size:
            self.full = True

    def get(self, length: int) -> np.ndarray:
        length = min(length, self.size if self.full else self.index)
        if length <= 0:
            return np.array([], dtype=np.float32)
        start = (self.index - length) % self.size
        if start < self.index:
            return self.buffer[start:self.index].copy()
        return np.concatenate((self.buffer[start:], self.buffer[:self.index])).copy()
