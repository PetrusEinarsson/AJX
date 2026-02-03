from abc import ABC, abstractmethod


class PreStepModifier(ABC):
    @abstractmethod
    def update_params(self, state, u, param):
        pass
