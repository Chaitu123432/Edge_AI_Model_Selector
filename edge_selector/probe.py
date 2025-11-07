# edge_selector/probe.py

import platform, psutil, json, shutil, subprocess, time

try:
    import pynvml
except ImportError:
    pynvml = None

class DeviceProbe:
    """Collects hardware and thermal information about the device."""

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
        return {'ram_total_mb': int(vm.total / (1024 * 1024))}

    def probe_gpu(self):
        info = {'gpus': []}
        if pynvml is None:
            print("⚠️ pynvml not installed – skipping GPU probe.")
            return info

        try:
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                try:
                    name = name.decode()
                except Exception:
                    pass
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                info['gpus'].append({
                    'name': name,
                    'mem_total_mb': int(mem.total / (1024 * 1024))
                })
            pynvml.nvmlShutdown()
        except pynvml.NVMLError_LibraryNotFound:
            print("⚠️ NVML not found – skipping GPU probe.")
        except Exception as e:
            print(f"⚠️ GPU probe error: {e}")
        return info

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

if __name__ == "__main__":
    profile = DeviceProbe().run()
    json.dump(profile, open('device_profile.json', 'w'), indent=2)
    print("✅ Device profile saved to device_profile.json")
