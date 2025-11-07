from edge_selector.probe import DeviceProbe
from edge_selector.dataset import DatasetBuilder
from edge_selector.benchmark import BenchmarkHarness
from edge_selector.model_registry import MODEL_REGISTRY

print("🚀 Starting full model benchmarking pipeline...")

device_prof = DeviceProbe().run()
builder = DatasetBuilder('runs.ndjson')

for name, info in MODEL_REGISTRY.items():
    print(f"\n⏳ Running {name} ({info['task']} - {info['framework']}) ...")
    model_obj, input_shape = info['load_fn']()
    bench = BenchmarkHarness(model_obj, info['framework'])
    result = bench.run(input_shape=input_shape)
    builder.append(result, device_prof, {'task': info['task'], 'model_name': name})
    print(f"✅ Done: {name} - Avg latency {result['latency_ms']:.2f} ms")

print("\\n🎯 Benchmarking complete! Results saved to runs.ndjson")
