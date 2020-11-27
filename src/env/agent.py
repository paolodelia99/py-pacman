from abc import ABC, abstractmethod


class Agent(ABC):
    name = 'agent'

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def act(self, state, **kwargs):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass
