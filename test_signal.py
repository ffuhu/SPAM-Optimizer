# to catch Ctrl+C and clean the resources gracefully
import signal
import sys

def graceful_exit(signum, frame):
    print(f"Received signal {signum}, exiting gracefully.")
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_exit)
signal.signal(signal.SIGINT, graceful_exit)

# Simulate a long-running process
while True:
    pass
