"""Small in-memory store for recent results that clients download after the response."""
import threading
import uuid
from collections import OrderedDict


class ResultCache:
    """The most recent results, kept in memory for download; the oldest are dropped first.

    Pinned results (the pre-computed sample results, backend/services/samples.py) are kept for the
    life of the process and do not count towards `size`.
    """

    def __init__(self, size):
        self.size = size
        self._items = OrderedDict()
        self._pinned = {}
        self._lock = threading.Lock()

    def put(self, data, key=None):
        key = key or uuid.uuid4().hex
        with self._lock:
            self._items[key] = data
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return key

    def pin(self, key, data):
        with self._lock:
            self._pinned[key] = data

    def get(self, key):
        with self._lock:
            return self._pinned[key] if key in self._pinned else self._items.get(key)
