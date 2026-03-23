import time
import sys

class ConsoleProgress:
    def __init__(self, width=30):
        self.width = width

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
