from abc import ABC, abstractmethod

class Actuator(ABC):
    @abstractmethod
    def apply(self, command) -> None: ...
    
    @abstractmethod
    def reset(self) -> None: ...