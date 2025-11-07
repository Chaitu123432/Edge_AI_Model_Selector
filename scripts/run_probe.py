from edge_selector.probe import DeviceProbe
import json
profile = DeviceProbe().run()
json.dump(profile, open('device_profile.json','w'), indent=2)
print('✅ Device profile saved!')