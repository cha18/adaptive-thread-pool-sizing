## ATPS

Regression model is used to find the optimal N (threadpool size).

This is tested on a task scheduler.
The task scheduler runs 3 task types:
1. CPU-Bound tasks
2. I/O-Bound tasks
3. Hybrid tasks

Demonstrations will run under 4 scenarios:
1. 100 I/O-bound tasks submitted immediately
2. 100 CPU-bound tasks submitted immediately
3. Rapid load spike - 1 task/sec then jump to 20 task/sec (50% I/O-bound, 50% CPU-bound)
4. Mixed spikes - Random I/O:CPU ratio every random number of seconds

Baseline program uses a static N (threadpool size).

The regression model's feature vector:
- Queue  (no. queued)
- CPU usage
- Thread wait time
- Thread service time
- No. of tasks completed in last second
- Current threadpool size
