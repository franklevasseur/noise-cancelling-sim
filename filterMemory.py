from typing import List

import numpy as np


class FilterMemory:

    p: int = 0
    memories: List[List[float]] = []

    def __init__(self, p: int):

        for i in range(p):
            self.memories.append([])

        self.p = p

    def append(self, h: np.ndarray):

        for i in range(self.p):
            self.memories[i].append(h[i])

    def getAll(self) -> List[List[float]]:
        return self.memories

