# src/utils/performance.py

###
### Functions that measure an extensive set of computational performance metrics
###

import time
import psutil
import torch
import pynvml # Importa a biblioteca de monitoramento da NVIDIA

class PerformanceMonitor:
    """
    Monitors a comprehensive set of computational performance metrics for a process.
    This version includes detailed GPU metrics like power, temperature, and utilization.
    """

    def __init__(self, gpu_index: int = 0):
        self.start_time = None
        self.process = psutil.Process()
        self.gpu_handle = None

        # Inicializa o NVML para monitoramento da GPU
        if torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except pynvml.NVMLError as error:
                print(f"Erro ao inicializar NVML: {error}. As métricas de GPU (energia, temp) não estarão disponíveis.")
                self.gpu_handle = None

    def start(self):
        """Starts the monitoring timers and records initial resource usage."""
        # Reseta as estatísticas de pico de memória da GPU
        if self.gpu_handle:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Captura os valores iniciais dos contadores
        self.disk_io_start = psutil.disk_io_counters()
        self.process.cpu_percent(interval=None) # Inicia o contador de CPU
        
        self.start_time = time.perf_counter()

    def stop(self) -> dict:
        """
        Stops the timers and returns a dictionary of all collected metrics.
        """
        if self.gpu_handle:
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()

        # --- Coleta de Todas as Métricas ---

        # 1. Tempo
        elapsed_time = end_time - self.start_time
        
        # 2. CPU
        cpu_usage = self.process.cpu_percent(interval=None) / psutil.cpu_count()
        
        # 3. RAM
        ram_peak_usage_bytes = self.process.memory_info().rss
        
        # 4. Disco I/O
        disk_io_end = psutil.disk_io_counters()
        disk_read_bytes = disk_io_end.read_bytes - self.disk_io_start.read_bytes
        disk_write_bytes = disk_io_end.write_bytes - self.disk_io_start.write_bytes

        # --- Métricas Detalhadas da GPU (se disponível) ---
        vram_peak_usage_bytes = 0
        gpu_power_watts = 0
        gpu_temp_celsius = 0
        gpu_utilization_percent = 0

        if self.gpu_handle:
            vram_peak_usage_bytes = torch.cuda.max_memory_allocated()
            gpu_power_watts = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convertido para Watts
            gpu_temp_celsius = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            gpu_utilization_percent = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu

        return {
            # Métricas Principais
            "elapsed_time_seconds": elapsed_time,
            "cpu_usage_percent": cpu_usage,
            "ram_peak_usage_bytes": ram_peak_usage_bytes,
            "vram_peak_usage_bytes": vram_peak_usage_bytes,
            
            # Métricas Detalhadas da GPU
            "gpu_power_watts": gpu_power_watts,
            "gpu_temp_celsius": gpu_temp_celsius,
            "gpu_utilization_percent": gpu_utilization_percent,
            
            # Métricas de I/O
            "disk_read_bytes": disk_read_bytes,
            "disk_write_bytes": disk_write_bytes,
        }

    def shutdown(self):
        """Shuts down the NVML library to release resources."""
        if self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as error:
                print(f"Erro ao finalizar NVML: {error}")
