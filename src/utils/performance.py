# src/utils/performance.py


###
### Functions that measure computational performance metrics
###


# imports
import time
import psutil
import torch
import pynvml # Para métricas mais detalhadas da GPU

# Inicializa o NVML para monitoramento da GPU
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assumindo GPU 0


# implementations

class PerformanceMonitor:
    """
    A utility class to monitor computational performance (time, CPU, RAM, VRAM).
    """
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
        # CPU
        self.cpu_usage_start = None
        
        # Memória RAM
        self.ram_usage_start = None
        
        # Memória VRAM (GPU)
        self.vram_usage_start = None

    def start(self):
        """Starts the monitoring timers and records initial resource usage."""
        self.cpu_usage_start = psutil.cpu_percent(interval=None)
        self.ram_usage_start = psutil.virtual_memory().used
        if torch.cuda.is_available():
            self.vram_usage_start = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            
        self.start_time = time.perf_counter()

    def stop(self) -> dict:
        """
        Stops the timers, records final resource usage, and returns a dictionary of metrics.
        """
        self.end_time = time.perf_counter()
        
        # Captura final do uso de recursos
        final_cpu_usage = psutil.cpu_percent(interval=None)
        final_ram_usage = psutil.virtual_memory().used
        final_vram_usage = 0
        if torch.cuda.is_available():
            final_vram_usage = pynvml.nvmlDeviceGetMemoryInfo(handle).used

        # Calcula as métricas
        elapsed_time = self.end_time - self.start_time
        cpu_usage = final_cpu_usage - self.cpu_usage_start
        ram_usage_bytes = final_ram_usage - self.ram_usage_start
        vram_usage_bytes = 0
        if self.vram_usage_start is not None:
             vram_usage_bytes = final_vram_usage - self.vram_usage_start

        return {
            "elapsed_time_seconds": elapsed_time,
            "cpu_usage_percent": cpu_usage,
            "ram_usage_bytes": ram_usage_bytes,
            "vram_usage_bytes": vram_usage_bytes
        }

# Lembre-se de chamar pynvml.nvmlShutdown() no final do seu script principal
# para liberar os recursos do NVML.
