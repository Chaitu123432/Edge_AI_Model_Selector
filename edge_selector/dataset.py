import json, os
class DatasetBuilder:
    def __init__(self, out_path='runs.ndjson'):
        self.out = out_path
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    def append(self, run_result, device_profile, meta):
        row = {**run_result, 'device': device_profile, 'meta': meta}
        with open(self.out, 'a') as f:
            f.write(json.dumps(row)+'\n')