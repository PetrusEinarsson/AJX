from abc import ABC, abstractmethod


class Sensor(ABC):

    @abstractmethod
    def observe(self, state, qdot_next, param):
        pass

    @abstractmethod
    def residual(self, target, prediction):
        pass
