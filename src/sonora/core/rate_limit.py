import threading
import time


class RateLimiter:
    """Thread-safe rate limiter with precise target_time scheduling."""

    _disabled: bool = False

    def __init__(self, interval_seconds: float) -> None:
        self.interval = interval_seconds
        self.lock = threading.Lock()
        self.last_call = 0.0

    @classmethod
    def set_disabled(cls, disabled: bool) -> None:
        cls._disabled = disabled

    def wait(self, interval_seconds: float | None = None) -> float:
        if self._disabled:
            return 0.0
        interval = self.interval if interval_seconds is None else interval_seconds
        if interval <= 0:
            return 0.0
        with self.lock:
            now = time.monotonic()
            target_time = max(now, self.last_call + interval)
            sleep_time = target_time - now
            self.last_call = target_time

        if sleep_time > 0:
            time.sleep(sleep_time)
        return sleep_time
