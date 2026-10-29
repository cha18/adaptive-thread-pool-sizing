from concurrent.futures import Future
import queue
from concurrent.futures import Executor
from collections.abc import Callable
import threading

_STOP = object()

class AdaptiveThreadPoolExecutor(Executor):
    def __init__(self, initial_workers, max_workers=64):
        if initial_workers > max_workers or initial_workers < 1: raise ValueError("initial_workers must be between 1 and max_workers")
        self._max_workers = max_workers
        self._work_queue = queue.Queue()
        self._live_threads = 0
        self._pills_queued = 0
        self._shutdown = False
        self._threads: set[threading.Thread] = set()

        self._lock = threading.Lock()
        self.resize(initial_workers)

    @property
    def num_workers(self):
        with self._lock:
            return self._live_threads - self._pills_queued

    def resize(self, new_size):
        if self._shutdown: return
        with self._lock:
            new_size = max(min(self._max_workers, new_size), 1)
            current_size = self._live_threads - self._pills_queued #num_workers is calculated manually without num_workers() since resize already holds the lock, so running num_workers() will cause a deadlock
            # a method that holds the lock must never call another method that takes the same lock - it will cause a deadlock
            if new_size < current_size:
                for _ in range(current_size-new_size):
                    self._work_queue.put(_STOP)
                    self._pills_queued += 1
            if new_size > current_size:
                for _ in range(new_size-current_size):
                    thread = threading.Thread(target=self._worker) #non-daemon: work is never cut off at exit; callers must use with shutdown()
                    self._threads.add(thread)
                    thread.start()
                    self._live_threads += 1

    def submit(self, fn: Callable, /, *args, **kwargs) -> Future:
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot submit to queue, workers are shutting down")
            future  = Future()
            self._work_queue.put((future, fn, args, kwargs))
        return future

    def shutdown(self, wait=True, *, cancel_futures=False) -> None:
        if cancel_futures: raise NotImplementedError("cancel futures is not supported")
        with self._lock:
            self._shutdown = True
            current_size = self._live_threads - self._pills_queued #same deadlock prevention mechanism as in resize()
            for _ in range(current_size):
                self._work_queue.put(_STOP)
                self._pills_queued += 1
        if wait:
            with self._lock:
                threads = list(self._threads) #clone threads to prevent the list from changing while being pilled
            for thread in threads:
                thread.join()

    def _worker(self):
        while True:
            item = self._work_queue.get()
            if item is _STOP:
                with self._lock:
                    self._threads.remove(threading.current_thread())
                    self._live_threads -= 1
                    self._pills_queued -= 1
                return
            else:
                future, fn, args, kwargs = item
                if not future.set_running_or_notify_cancel(): continue
                try:
                    result = fn(*args, **kwargs)
                except BaseException as e:
                    future.set_exception(e)
                else:
                    future.set_result(result)
