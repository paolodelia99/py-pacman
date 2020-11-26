from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def act(self, **kwargs):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass
