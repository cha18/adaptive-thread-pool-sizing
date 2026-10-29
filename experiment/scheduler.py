import time
import numpy as np
import random
import psutil
from concurrent.futures import ThreadPoolExecutor

# Reusing your task logic
class Tasks:
    @staticmethod
    def cpu_intensive_task(size_multiplier):
        start = time.perf_counter()
        m1 = np.random.rand(5*size_multiplier, 5*size_multiplier)
        m2 = np.random.rand(5*size_multiplier, 5*size_multiplier)
        np.multiply(m1, m2)
        return time.perf_counter() - start

    @staticmethod
    def io_bound_task(io_delay):
        start = time.perf_counter()
        time.sleep(io_delay) 
        return time.perf_counter() - start

class TaskSchedulerEnv:
    def __init__(self):
        # Action Space: 0=Decrease N, 1=Hold, 2=Increase N
        self.action_dim = 3
        # State: [QueueSize, CPU%, WaitTime, ServiceTime, Completed/s, Current_N]
        self.state_dim = 6
        
        self.min_n = 1
        self.max_n = 64
        self.current_n = 4 # Start small
        
        self.pending_queue = [] # List of (type, params)
        self.active_tasks = 0
        self.executor = ThreadPoolExecutor(max_workers=100) # Physical limit
        
        # Metrics storage
        self.completed_history = []
        self.wait_time_history = []
        self.service_time_history = []
        
        self.start_time = time.time()
        self.last_step_time = time.time()

    def reset(self):
        self.current_n = 4
        self.pending_queue = []
        self.active_tasks = 0
        self.completed_history = []
        self.wait_time_history = []
        self.service_time_history = []
        self.last_step_time = time.time()
        return self._get_state()

    def _generate_traffic(self, scenario):
        # Scenario logic to add tasks to pending_queue
        # 0: IO Heavy, 1: CPU Heavy, 2: Mixed
        num_new = 0
        if scenario == 'spike':
            num_new = 20
        else:
            num_new = random.randint(1, 5)

        for _ in range(num_new):
            if scenario == 'cpu':
                # Type 1: CPU
                self.pending_queue.append(('cpu', random.randint(50, 100)))
            elif scenario == 'io':
                # Type 2: IO
                self.pending_queue.append(('io', random.uniform(0.05, 0.15)))
            else:
                # Mixed
                if random.random() > 0.5:
                    self.pending_queue.append(('cpu', random.randint(50, 100)))
                else:
                    self.pending_queue.append(('io', random.uniform(0.05, 0.15)))

    def step(self, action, scenario_type):
        # 1. Apply Action (Adjust ThreadPool Size N)
        if action == 0: # Decrease
            self.current_n = max(self.min_n, self.current_n - 1)
        elif action == 2: # Increase
            self.current_n = min(self.max_n, self.current_n + 1)
        
        # 2. Simulate Incoming Traffic
        self._generate_traffic(scenario_type)
        
        # 3. Process Tasks (Move from Pending -> Active based on N)
        # We simulate a time step of 0.1 seconds
        step_duration = 0.1
        start_wait = time.time()
        
        tasks_launched = 0
        
        # Submit tasks if slots are open
        while len(self.pending_queue) > 0 and self.active_tasks < self.current_n:
            task_type, param = self.pending_queue.pop(0)
            self.active_tasks += 1
            tasks_launched += 1
            
            # Add wait time (Time in queue)
            self.wait_time_history.append(time.time() - start_wait)
            
            # Submit to real thread pool
            if task_type == 'cpu':
                f = self.executor.submit(Tasks.cpu_intensive_task, param)
            else:
                f = self.executor.submit(Tasks.io_bound_task, param)
            
            f.add_done_callback(self._task_complete_callback)

        time.sleep(step_duration) # Wait for step to pass
        
        # 4. Calculate Reward
        # Reward = Throughput - (Penalty * QueueBuildup)
        throughput = len(self.completed_history) / step_duration
        queue_len = len(self.pending_queue)
        
        # We want high throughput, low queue
        reward = throughput - (0.1 * queue_len)
        
        # Metrics for State
        state = self._get_state(throughput)
        
        # Clear instantaneous history for next step
        self.completed_history = []
        
        return state, reward

    def _task_complete_callback(self, future):
        self.active_tasks -= 1
        try:
            res = future.result()
            self.service_time_history.append(res)
            self.completed_history.append(1)
        except:
            pass

    def _get_state(self, current_throughput=0):
        # [Queue, CPU, Wait, Service, Throughput, N]
        cpu_usage = psutil.cpu_percent()
        
        avg_wait = np.mean(self.wait_time_history[-20:]) if self.wait_time_history else 0
        avg_service = np.mean(self.service_time_history[-20:]) if self.service_time_history else 0
        
        return np.array([
            len(self.pending_queue),
            cpu_usage,
            avg_wait,
            avg_service,
            current_throughput,
            self.current_n
        ], dtype=np.float32)