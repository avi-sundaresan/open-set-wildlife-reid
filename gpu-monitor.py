import pynvml
import time
import schedule
import os

# Helper script that runs the desired script when the GPU usage dips below a certain threshold

def get_min_gpu_usage():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    min_utilization = float('inf')  # Initialize with infinity

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        print(f"GPU {i}: {utilization.gpu}%")
        if utilization.gpu < min_utilization:
            min_utilization = utilization.gpu

    pynvml.nvmlShutdown()
    return min_utilization

def check_and_run_script():
    gpu_usage = get_min_gpu_usage()

    if gpu_usage < THRESHOLD: 
        print("Running script...")
        os.system("python3 eval.py")

# Usage threshold
THRESHOLD = 20
# Schedule the check
schedule.every(2).minutes.do(check_and_run_script)

while True:
    schedule.run_pending()
    time.sleep(5)
    
