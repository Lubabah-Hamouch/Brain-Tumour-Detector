"""Small in-memory store for recent results that clients download after the response."""
import threading
import uuid
from collections import OrderedDict


class ResultCache:
    """The most recent results, kept in memory for download; the oldest are dropped first."""

    def __init__(self, size):
        self.size = size
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def put(self, data):
        key = uuid.uuid4().hex
        with self._lock:
            self._items[key] = data
            while len(self._items) > self.size:
                self._items.popitem(last=False)
        return key

    def get(self, key):
        with self._lock:
            return self._items.get(key)
