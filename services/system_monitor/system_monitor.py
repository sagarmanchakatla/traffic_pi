import subprocess
import psutil

TEMP_THRESHOLD = 75
CPU_THRESHOLD = 80

class SystemMonitor:
    """Monitor Pi temperature and performance"""
    
    @staticmethod
    def get_cpu_temp():
        try:
            result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                  capture_output=True, text=True, timeout=1)
            temp_str = result.stdout.strip()
            temp = float(temp_str.split('=')[1].split("'")[0])
            return temp
        except:
            return 0.0
    
    @staticmethod
    def get_cpu_usage():
        return psutil.cpu_percent(interval=0.5)
    
    @staticmethod
    def should_throttle(temp, cpu):
        return temp > TEMP_THRESHOLD or cpu > CPU_THRESHOLD
    
    @staticmethod
    def log_stats(temp, cpu, memory):
        status = "⚠️  THROTTLING" if SystemMonitor.should_throttle(temp, cpu) else "✓ OK"
        print(f"[SYS] {status} | Temp: {temp:.1f}°C | CPU: {cpu:.1f}% | RAM: {memory:.1f}%")
