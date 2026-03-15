
import subprocess
import psutil
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value
from ctypes import c_bool 
import time

from services.system_monitor.system_monitor import SystemMonitor

def system_monitor_process(
    emergency_event: Event,
    throttle_mode: Value,
    status_queue: Queue,
    monitor_interval: int
):
    """
    System monitoring process
    Monitors temperature, CPU, RAM and sets throttle mode
    """
    process_name = "MONITOR"
    print(f"[{process_name}] Process started (PID: {mp.current_process().pid})")
    print(f"[{process_name}] Monitor interval: {monitor_interval}s")
    
    while not emergency_event.is_set():
        time.sleep(monitor_interval)
        
        if emergency_event.is_set():
            break
        
        # Get system stats
        temp = SystemMonitor.get_cpu_temp()
        cpu = SystemMonitor.get_cpu_usage()
        memory = psutil.virtual_memory().percent
        
        # Update throttle mode
        should_throttle = SystemMonitor.should_throttle(temp, cpu)
        throttle_mode.value = should_throttle
        
        # Log stats
        SystemMonitor.log_stats(temp, cpu, memory, process_name)
        
        # Send status
        status_queue.put({
            'process': 'monitor',
            'temp': temp,
            'cpu': cpu,
            'memory': memory,
            'throttling': should_throttle
        })
    
    print(f"[{process_name}] Process stopped")
 
 