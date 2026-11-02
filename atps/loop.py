import threading


class ControlLoop:
    def __init__(self, pool, controller, interval=0.1):
        self._pool = pool
        self._controller = controller
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()  # sets the flag which wakes the Event.wait at once. stop() never has to wait a full interval
        if self._thread.is_alive():
            self._thread.join()

    def _run(self):
        while not self._stop.wait(self._interval): #self._stop.wait(_interval) blocks for _interval time. If the block times out (nobody set the flag), it returns False. It will return True if somebody does _stop.set() before the time out (somebody set the flag)
            snapshot = self._pool.stats.snapshot()
            target = self._controller.decide(snapshot, self._pool.num_workers, self._pool.queue_length)
            self._pool.resize(target)
