from settings import *
from collections import deque
import numpy as np
import random


class RLMemory:
    def __init__(self):
        self.buffer_size = BUFFER_SIZE
        self.count = 0
        self.buffer = deque()

    def add(self, a, r, ja):
        exp = (a, r, ja)
        if self.count < self.buffer_size:
            self.buffer.append(exp)
            self.count += 1
        else:
            j = random.randrange(1, self.buffer_size + 1)
            if j < self.buffer_size:
                self.buffer[j] = exp

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def size(self):
        return self.count

    def sample(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        a_batch = np.array([_[0] for _ in batch])
        r_batch = np.array([_[1] for _ in batch])
        joint_act_batch = np.array([_[2] for _ in batch])
        return a_batch, r_batch, joint_act_batch


class CentralAgentMemory:
    def __init__(self):
        self.buffer_size = BUFFER_SIZE
        self.count = 0
        self.buffer = deque()

    def add(self, a, r):
        exp = (a, r)
        if self.count < self.buffer_size:
            self.buffer.append(exp)
            self.count += 1
        else:
            j = random.randrange(1, self.buffer_size + 1)
            if j < self.buffer_size:
                self.buffer[j] = exp

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def size(self):
        return self.count

    def sample(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        a_batch = np.array([_[0] for _ in batch])
        r_batch = np.array([_[1] for _ in batch])
        return a_batch, r_batch
