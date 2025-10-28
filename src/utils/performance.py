# src/utils/performance.py

import time
import threading
from typing import Dict, List, Optional

import psutil
import torch

try:
    import pynvml as nvml
    NVML_AVAILABLE = True
except Exception:
    nvml = None
    NVML_AVAILABLE = False


class PerformanceMonitor:
    """
    Monitora métricas de performance com amostragem contínua de GPU em uma thread separada,
    agregando médias e picos de potência, temperatura, utilização e memória global da GPU.
    """

    def __init__(self, gpu_index: Optional[int] = 0, sample_interval: float = 0.1):
        self.start_time = None
        self.process = psutil.Process()
        self.sample_interval = sample_interval

        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()

        # NVML / GPU
        self.gpu_index = gpu_index
        self.gpu_handle = None

        # Buffers de amostras
        self.gpu_power_samples: List[float] = []       # Watts
        self.gpu_temp_samples: List[int] = []          # °C
        self.gpu_util_samples: List[int] = []          # %
        self.gpu_mem_used_samples: List[int] = []      # bytes (NVML used)

        # Inicializa NVML se disponível e CUDA presente
        if NVML_AVAILABLE and torch.cuda.is_available() and self.gpu_index is not None:
            try:
                nvml.nvmlInit()
                if self.gpu_index is None:
                    self.gpu_index = torch.cuda.current_device()
                self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(int(self.gpu_index))
            except Exception as e:
                print(f"Erro ao inicializar NVML: {e}. Métricas de GPU indisponíveis.")
                self.gpu_handle = None

    def _monitor_gpu(self):
        """Thread: coleta potência (W), temperatura (°C), utilização (%) e memória usada (bytes)."""
        while not self._stop_monitoring.is_set():
            if not self.gpu_handle:
                break
            # Potência (mW -> W)
            try:
                power_w = nvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
            except Exception:
                power_w = 0.0
            # Temperatura (°C)
            try:
                temp_c = nvml.nvmlDeviceGetTemperature(self.gpu_handle, nvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp_c = 0
            # Utilização (%)
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
            except Exception:
                util = 0
            # Memória global usada (bytes)
            try:
                mem = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                mem_used = int(mem.used)
            except Exception:
                mem_used = 0

            self.gpu_power_samples.append(float(power_w))
            self.gpu_temp_samples.append(int(temp_c))
            self.gpu_util_samples.append(int(util))
            self.gpu_mem_used_samples.append(int(mem_used))

            time.sleep(self.sample_interval)

    def start(self):
        """Inicia contadores e thread de monitoramento da GPU."""
        # Preparação CPU/Disco/Tempo
        self.process.cpu_percent(interval=None)
        self.disk_io_start = psutil.disk_io_counters()
        self.start_time = time.perf_counter()

        # Reset de pico do PyTorch no device alvo
        if self.gpu_handle is not None:
            dev = self.gpu_index if self.gpu_index is not None else torch.cuda.current_device()
            torch.cuda.synchronize(dev)
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats(dev)
            else:
                torch.cuda.memory.reset_peak_memory_stats(dev)

            # Limpa buffers e inicia a thread
            self.gpu_power_samples.clear()
            self.gpu_temp_samples.clear()
            self.gpu_util_samples.clear()
            self.gpu_mem_used_samples.clear()
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(target=self._monitor_gpu, daemon=True)
            self._monitoring_thread.start()

    def stop(self) -> Dict[str, float]:
        """Para contadores e thread; retorna dicionário com métricas agregadas (avg/peak)."""
        if self.gpu_handle is not None:
            dev = self.gpu_index if self.gpu_index is not None else torch.cuda.current_device()
            torch.cuda.synchronize(dev)
            self._stop_monitoring.set()
            if self._monitoring_thread:
                self._monitoring_thread.join()

        end_time = time.perf_counter()
        elapsed_time = end_time - (self.start_time or end_time)

        # CPU/RAM/Disco
        cpu_usage_norm = self.process.cpu_percent(interval=None)
        try:
            cpu_usage_norm = cpu_usage_norm / max(psutil.cpu_count(logical=True) or 1, 1)
        except Exception:
            pass

        ram_rss_bytes = float(self.process.memory_info().rss)
        disk_io_end = psutil.disk_io_counters()
        disk_read_bytes = float(disk_io_end.read_bytes - self.disk_io_start.read_bytes)
        disk_write_bytes = float(disk_io_end.write_bytes - self.disk_io_start.write_bytes)

        # Pico de VRAM do processo (PyTorch)
        if self.gpu_handle is not None and torch.cuda.is_available():
            dev = self.gpu_index if self.gpu_index is not None else torch.cuda.current_device()
            try:
                vram_peak_usage_bytes = float(torch.cuda.max_memory_allocated(dev))
            except Exception:
                vram_peak_usage_bytes = 0.0
        else:
            vram_peak_usage_bytes = 0.0

        # Agregadores
        def _avg(xs: List[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        def _peak(xs: List[float]) -> float:
            return float(max(xs)) if xs else 0.0

        # Snapshot final da VRAM global (para compatibilidade)
        gpu_vram_used_bytes_snapshot = 0.0
        try:
            if self.gpu_handle is not None:
                mem = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_vram_used_bytes_snapshot = float(mem.used)
        except Exception:
            pass

        return {
            "elapsed_time_seconds": float(elapsed_time),

            # CPU/RAM/Disk
            "cpu_usage_percent": float(cpu_usage_norm),
            "ram_peak_usage_bytes": float(ram_rss_bytes),
            "disk_read_bytes": float(disk_read_bytes),
            "disk_write_bytes": float(disk_write_bytes),

            # PyTorch (processo)
            "vram_peak_usage_bytes": float(vram_peak_usage_bytes),

            # GPU Power (W)
            "gpu_power_watts_avg": _avg(self.gpu_power_samples),
            "gpu_power_watts_peak": _peak(self.gpu_power_samples),

            # GPU Temp (°C)
            "gpu_temp_celsius_avg": _avg(self.gpu_temp_samples),
            "gpu_temp_celsius_peak": _peak(self.gpu_temp_samples),

            # GPU Utilization (%)
            "gpu_utilization_percent_avg": _avg(self.gpu_util_samples),
            "gpu_utilization_percent_peak": _peak(self.gpu_util_samples),

            # GPU Memory global via NVML (bytes) — amostrado
            "gpu_mem_used_bytes_avg": _avg(self.gpu_mem_used_samples),
            "gpu_mem_used_bytes_peak": _peak(self.gpu_mem_used_samples),

            # Snapshot final (compatibilidade)
            "gpu_vram_used_bytes": float(gpu_vram_used_bytes_snapshot),
        }

    def shutdown(self):
        """Finaliza NVML (se ativo)."""
        if NVML_AVAILABLE and self.gpu_handle is not None:
            try:
                nvml.nvmlShutdown()
            except Exception as e:
                print(f"Erro ao finalizar NVML: {e}")
