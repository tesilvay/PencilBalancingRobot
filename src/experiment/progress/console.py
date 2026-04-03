from dataclasses import dataclass
import time
import sys

from .base import Progress


@dataclass
class ProgressParams:
    width: int


PROGRESS_PRESETS = {
    "default": {
        "width": 30,
    }
}


class ConsoleProgress(Progress):
    def __init__(self, params : ProgressParams):
        self.width = params.width

    def start(self, total, label=""):
        self.total = total
        self.label = label
        self.start_time = time.time()

    def update(self, step):
        progress = step / self.total
        filled = int(self.width * progress)

        bar = "█" * filled + "-" * (self.width - filled)
        percent = progress * 100

        elapsed = time.time() - self.start_time

        sys.stdout.write(
            f"\r{self.label} |{bar}| {percent:6.2f}% "
            f"({step}/{self.total}) {elapsed:5.1f}s"
        )
        sys.stdout.flush()

    def finish(self):
        total_time = time.time() - self.start_time
        sys.stdout.write(f"  done in {total_time:.2f}s\n")
        sys.stdout.flush()
