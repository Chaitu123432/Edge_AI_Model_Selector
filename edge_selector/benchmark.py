import time, psutil, numpy as np, json, uuid, torch, onnxruntime as ort

class BenchmarkHarness:
    """
    Handles both Torch and ONNX models, automatically infers type.
    """
    def __init__(self, model_obj, framework, device='cpu'):
        self.framework = framework
        self.device = device
        self.model = model_obj
        if framework == 'torch':
            self.model.eval()
            if torch.cuda.is_available() and device == 'cuda':
                self.model.to('cuda')

    def run(self, input_shape=(1,3,224,224), runs=20):
        x = np.random.rand(*input_shape).astype(np.float32)
        latencies = []

        for _ in range(runs):
            start = time.time()

            if self.framework == 'onnx':
                inp = {self.model.get_inputs()[0].name: x}
                self.model.run(None, inp)
            else:  # torch
                t = torch.from_numpy(x)
                if self.device == 'cuda' and torch.cuda.is_available():
                    t = t.cuda()
                with torch.no_grad():
                    _ = self.model(t)
                if self.device == 'cuda':
                    torch.cuda.synchronize()

            latencies.append((time.time() - start)*1000)

        mem_mb = psutil.Process().memory_info().rss / (1024*1024)
        return {
            'run_id': str(uuid.uuid4()),
            'framework': self.framework,
            'latency_ms': float(np.mean(latencies)),
            'latency_p90': float(np.percentile(latencies, 90)),
            'mem_mb': mem_mb,
            'device': self.device
        }
