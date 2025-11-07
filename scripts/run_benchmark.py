from edge_selector.probe import DeviceProbe
from edge_selector.benchmark import BenchmarkHarness
from edge_selector.dataset import DatasetBuilder

prof = DeviceProbe().run()
bench = BenchmarkHarness('model.onnx')
res = bench.run()
DatasetBuilder('runs.ndjson').append(res, prof, {'task':'classification'})
print('✅ Benchmark completed!')