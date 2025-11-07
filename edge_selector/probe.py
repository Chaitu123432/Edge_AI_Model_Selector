import platform, psutil, json, shutil, subprocess, time

try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

class DeviceProbe:
    def probe_cpu(self):
        freq = psutil.cpu_freq()
        return {
            'processor': platform.processor() or platform.machine(),
            'cores_logical': psutil.cpu_count(logical=True),
            'cores_physical': psutil.cpu_count(logical=False),
            'freq_max_mhz': getattr(freq, 'max', None)
        }

    def probe_ram(self):
        vm = psutil.virtual_memory()
        return {'ram_total_mb': vm.total // (1024 * 1024)}

    def probe_gpu(self):
        if not PYNVML_AVAILABLE:
            return {'gpus': []}
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpus.append({
                'name': pynvml.nvmlDeviceGetName(h).decode(),
                'mem_total_mb': pynvml.nvmlDeviceGetMemoryInfo(h).total // (1024 * 1024)
            })
        return {'gpus': gpus}

    def probe_thermals(self):
        temps = {}
        try:
            for k, v in psutil.sensors_temperatures().items():
                temps[k] = [{'label': e.label, 'current': e.current} for e in v]
        except Exception:
            pass
        return temps

    def run(self):
        return {
            'timestamp': time.time(),
            'os': platform.system(),
            'cpu': self.probe_cpu(),
            'ram': self.probe_ram(),
            'gpu': self.probe_gpu(),
            'thermals': self.probe_thermals()
        }