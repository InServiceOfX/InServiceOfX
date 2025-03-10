from threading import Lock
from io import StringIO

class ThreadSafeStringBuilder:
    """
    Make a thread safe dynamically allocated string.
    """
    def __init__(self):
        self._lock = Lock()
        self._buffer = StringIO()

    def append(self, text: str) -> None:
        with self._lock:
            self._buffer.write(text)

    def get_value(self) -> str:
        with self._lock:
            return self._buffer.getvalue()
