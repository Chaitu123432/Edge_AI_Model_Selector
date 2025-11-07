from edge_selector.surrogate import Surrogate
s = Surrogate()
s.train('runs.ndjson')
print('✅ Surrogate trained!')