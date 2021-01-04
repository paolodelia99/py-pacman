from abc import ABC, abstractmethod


class Agent(ABC):
    """
    A class representing an AI agent
    """
    name = 'agent'

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def act(self, *args, **kwargs):
        """
        Given the state the agent return the action to take

        :param args: arguments
        :param kwargs: other argument passed by keyword
        :return: the action to take
        """
        pass

    @abstractmethod
    def train(self, **kwargs):
        """
        Train the agent

        :param kwargs:
        :return:
        """
        pass
