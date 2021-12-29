from traceback import format_tb
from typing import Optional, Union, Protocol
from time import time
from enum import Enum, auto

from rich.console import Console
from rich.progress import GetTimeCallable, Progress, ProgressColumn


class ExpProgress(Protocol):
    def __init__(
        self,
        total_iters: int,
        *args,
        timeout: Optional[int] = None,
        early_stop: Optional[int] = None,
        **kwargs,
    ) -> None:
        ...

    def __enter__(self) -> "ExpProgress":
        ...

    def __exit__(self) -> None:
        ...

    def iteration_update(self) -> None:
        """
        Use only in context manager.
        """

        ...

    def reset_early_stop_cnt(self) -> None:
        """
        Use only in context manager.
        """

        ...


class ExpProgressSilent:
    def __init__(
        self,
        total_iters: int,
        timeout: Optional[int] = None,
        early_stop: Optional[int] = None,
    ) -> None:
        self.total_iters = total_iters
        self.timeout = timeout
        self.early_stop = early_stop

    def __enter__(self) -> "ExpProgressSilent":
        self.iter_n = 0
        early_stop = self.early_stop
        if early_stop is not None:
            self.to_early = early_stop
        timeout = self.timeout
        if timeout is not None:
            self.to_timeout = timeout
            self.t0 = time()
        return self

    def __exit__(self, *_) -> None:
        ...

    def iteration_update(self) -> None:
        """
        Use only in context manager.
        """

        self.iter_n += 1
        if self.iter_n >= self.total_iters:
            raise EndExperiment(EndReason.ITERATIONS)
        if self.early_stop is not None:
            t_delta = time() - self.t0
            self.to_timeout -= t_delta
            if self.to_timeout <= 0:
                raise EndExperiment(EndReason.TIMEOUT)
        if self.early_stop is not None:
            self.to_early -= 1
            if self.to_early <= 0:  # type: ignore
                raise EndExperiment(EndReason.EARLY_STOP)

    def reset_early_stop_cnt(self) -> None:
        """
        Use only in context manager.
        """

        if self.early_stop is not None:
            self.to_early = self.early_stop


class ExpProgressShow(Progress):
    def __init__(
        self,
        total_iters: int,
        *columns: Union[str, ProgressColumn],
        timeout: Optional[int] = None,
        early_stop: Optional[int] = None,
        console: Optional[Console] = None,
        auto_refresh: bool = True,
        refresh_per_second: float = 1,
        speed_estimate_period: float = 30,
        transient: bool = True,
        redirect_stdout: bool = True,
        redirect_stderr: bool = True,
        get_time: Optional[GetTimeCallable] = None,
        disable: bool = False,
        expand: bool = False,
    ) -> None:
        super().__init__(
            *columns,
            console=console,
            auto_refresh=auto_refresh,
            refresh_per_second=refresh_per_second,
            speed_estimate_period=speed_estimate_period,
            transient=transient,
            redirect_stdout=redirect_stdout,
            redirect_stderr=redirect_stderr,
            get_time=get_time,
            disable=disable,
            expand=expand,
        )
        self.total_iters = total_iters
        self.timeout = timeout
        self.early_stop = early_stop

    def __enter__(self) -> "ExpProgressShow":
        super().__enter__()
        self.iter_n = 0
        self.task_iters = self.add_task("[green]Iterations", total=self.total_iters)
        timeout = self.timeout
        early_stop = self.early_stop
        if early_stop is not None:
            self.task_early = self.add_task("[orange]To early stop", total=early_stop)
            self.to_early = early_stop
        else:
            self.task_early = None
        if timeout is not None:
            self.task_time = self.add_task("[blue]To timeout", total=timeout)
            self.to_timeout = timeout
            self.t0 = time()
        else:
            self.task_time = None
        return self

    def iteration_update(self) -> None:
        """
        Use only in context manager.
        """

        self.update(self.task_iters, advance=1)
        self.iter_n += 1
        if self.iter_n >= self.total_iters:
            raise EndExperiment(EndReason.ITERATIONS)
        task_time = self.task_time
        if task_time is not None:
            t_delta = time() - self.t0
            self.update(task_time, advance=t_delta)
            self.to_timeout -= t_delta
            if self.to_timeout <= 0:
                raise EndExperiment(EndReason.TIMEOUT)
        task_early = self.task_early
        if task_early is not None:
            self.update(task_early, advance=1)
            self.to_early -= 1
            if self.to_early <= 0:  # type: ignore
                raise EndExperiment(EndReason.EARLY_STOP)

    def reset_early_stop_cnt(self) -> None:
        """
        Use only in context manager.
        """

        if self.early_stop is not None:
            self.to_early = self.early_stop
            self.update(self.task_early, completed=0)  # type: ignore


class EndReason(Enum):
    ITERATIONS = auto()
    EARLY_STOP = auto()
    TIMEOUT = auto()
    EXCEPTION = auto()


class EndExperiment(Exception):
    def __init__(
        self, reason: EndReason, *args: object, exception: Optional[Exception] = None
    ) -> None:
        super().__init__(*args)
        self.reason = reason
        self.exception = exception

    def __str__(self) -> str:
        msg = f"{EndExperiment.__name__} reason: {self.reason.name.lower()}"
        e = self.exception
        print(f"{e = }")
        if e is not None:
            sep = ", "
            tb = "\n".join(format_tb(e.__traceback__))
            msg += f"\n\texception: {type(e)}: {sep.join(e.args)}\n\ttraceback: {tb}"
        return msg
