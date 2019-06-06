from collections import deque
import numpy as np


class Memory:

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]
