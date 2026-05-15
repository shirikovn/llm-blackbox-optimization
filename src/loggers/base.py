from abc import ABC
from abc import abstractmethod


class BaseLogger(ABC):
    @abstractmethod
    def log_prompt(
        self,
        step,
        prompt,
    ):
        pass

    @abstractmethod
    def log_response(
        self,
        step,
        response,
    ):
        pass

    @abstractmethod
    def log_step(
        self,
        step,
        x,
        fx,
        grad,
    ):
        pass

    @abstractmethod
    def log_metric(
        self,
        name,
        value,
    ):
        pass

    @abstractmethod
    def log_error(
        self,
        error,
    ):
        pass

    @abstractmethod
    def save_config(
        self,
        config,
    ):
        pass

    @abstractmethod
    def finalize(
        self,
    ):
        pass
