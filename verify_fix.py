"""Verify frame_017 fix"""
import json
from pathlib import Path

path = Path('outputs/analysis/frame_017_analysis.json')
with open(path, 'r') as f:
    data = json.load(f)

print('VERIFICATION - Frame 017 Analysis:')
print('=' * 50)
print(f"scene_type: {data['scene_type']}")
print(f"people_count: {data['people_count']}")
print(f"vlm_description: {data['vlm_description'][:70]}...")
print(f"activity: {data['activity'][:70]}...")
print(f"alert_priority_signals: {data['alert_priority_signals']}")
print(f"suspicious_elements: {data['suspicious_elements']}")
print(f"person_features: {len(data['person_features'])} profiles")
print(f"object_details: {len(data['object_details'])} objects")
print(f"recommended_action: {data['recommended_action']}")

# Check all fields are correct
all_correct = (
    data['scene_type'] == 'interior' and
    data['people_count'] == 5 and
    'retail counter' in data['vlm_description'] and
    data['alert_priority_signals'] == ['HIGH', 'suspicious_activity'] and
    len(data['suspicious_elements']) > 0 and
    len(data['person_features']) == 3 and
    len(data['object_details']) == 3
)

if all_correct:
    print('\n✅ ALL FIELDS CORRECT - Fix successful!')
else:
    print('\n❌ Some fields still incorrect')
