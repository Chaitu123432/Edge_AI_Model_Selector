import json
from edge_selector.surrogate import Surrogate
from edge_selector.selector import Selector

context = json.load(open('device_profile.json'))
actions = json.load(open('examples/actions.json'))

s = Surrogate()
s.train('runs.ndjson')
sel = Selector(s)
print(sel.select(context, actions))