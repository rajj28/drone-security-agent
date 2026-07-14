# Cursor AI Prompt: Test and Evaluate Drone Security Pipeline

## Context
You are testing a drone security surveillance system that analyzes video footage to detect security threats. The system uses:
- **Frame Extraction**: Intelligent extraction with quality filtering (rejects dark, blurry, text-overlay frames)
- **Telemetry Generation**: Motion detection, people counting, scene analysis
- **Vision Analysis**: CLIP (object detection) + BLIP (captioning) + GPT-4o (security assessment)
- **Alert Generation**: CRITICAL, HIGH, MEDIUM, LOW, CLEAR threat levels

## Current Issue
The test script fails with `ModuleNotFoundError: No module named 'pymongo'`. The system has MongoDB dependencies but pymongo is not installed.

## Task: Test and Evaluate the Pipeline

### Step 1: Fix Dependencies
1. Install pymongo: `pip install pymongo`
2. Or make MongoDB optional in the code (wrap imports in try/except)

### Step 2: Test Frame Extraction
Run frame extraction on the test video:
```bash
python -c "
from src.intelligent_frame_extractor import IntelligentFrameExtractor
from pathlib import Path
import sys

video_path = 'data/videos/Sneaky Thieves Caught Stealing Phones On Camera - Newsflare (1080p, h264) (1).mp4'
output_dir = Path('data/test_extraction')
output_dir.mkdir(parents=True, exist_ok=True)

extractor = IntelligentFrameExtractor()
frames = extractor.extract_hybrid(Path(video_path), str(output_dir), max_frames=10)

print(f'Extracted {len(frames)} frames')
for f in frames:
    print(f'  - {f.filename} at {f.timestamp:.2f}s: {f.extraction_reason}')
"
```

**Expected Output:**
- Should extract ~5-10 frames (some rejected by quality filter)
- Rejected frames: dark, blurry, text overlays
- Accepted frames: actual shop scenes with people

### Step 3: Test Vision Analysis on Single Frame
Test CLIP + BLIP + GPT-4o on one extracted frame:
```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.cloud_enhanced_analyzer import CloudEnhancedAnalyzer
from src.config import settings
import base64

# Use first extracted frame
frame_path = Path('data/test_extraction/frame_001.jpg')
if not frame_path.exists():
    print('Frame not found - run extraction first')
    sys.exit(1)

# Read and encode frame
with open(frame_path, 'rb') as f:
    frame_data = base64.b64encode(f.read()).decode()

# Analyze
analyzer = CloudEnhancedAnalyzer()
result = analyzer.analyze_frame(frame_data, frame_id='test_001')

print('=== VISION ANALYSIS RESULT ===')
print(f'Threat Level: {result.get(\"threat_level\", \"N/A\")}')
print(f'People Count: {result.get(\"people_count\", \"N/A\")}')
print(f'Scene Type: {result.get(\"scene_type\", \"N/A\")}')
print(f'Description: {result.get(\"description\", \"N/A\")[:200]}...')
print(f'Model Used: {result.get(\"model_used\", \"N/A\")}')

if result.get('alert'):
    alert = result['alert']
    print(f'\\n=== ALERT ===')
    print(f'Severity: {alert.get(\"severity\", \"N/A\")}')
    print(f'Reason: {alert.get(\"reason\", \"N/A\")}')
"
```

**Expected Output for Shop Theft Video:**
- **Threat Level**: CRITICAL or HIGH (not CLEAR!)
- **People Count**: 4-6 (not 1!)
- **Scene Type**: retail
- **Description**: Should mention people at counter, reaching, phones
- **Alert**: Should trigger for suspicious behavior

### Step 4: Evaluate Threat Detection Quality

Check if the system correctly identifies:
1. **Retail Theft Indicators**:
   - [ ] Hand reaching toward drawers/cabinets
   - [ ] Coordinated group (distraction tactic)
   - [ ] Multiple people crowding counter
   - [ ] Concealing items

2. **People Counting Accuracy**:
   - [ ] Counts ALL visible people (not just 1)
   - [ ] Includes people in background
   - [ ] Includes partially visible people

3. **Threat Level Assignment**:
   - [ ] Drawer reaching = CRITICAL
   - [ ] Coordinated group = CRITICAL
   - [ ] Loitering = MEDIUM
   - [ ] Normal shopping = CLEAR

### Step 5: Test Frame Quality Filtering
Verify that bad frames are rejected:
```bash
python -c "
from src.intelligent_frame_extractor import IntelligentFrameExtractor
import cv2
import numpy as np

extractor = IntelligentFrameExtractor()

# Test quality filter
test_cases = [
    ('dark_frame', np.zeros((100, 100, 3), dtype=np.uint8)),
    ('bright_frame', np.ones((100, 100, 3), dtype=np.uint8) * 255),
    ('solid_color', np.ones((100, 100, 3), dtype=np.uint8) * 128),
]

for name, frame in test_cases:
    is_ok, reason = extractor._is_frame_quality_acceptable(frame)
    print(f'{name}: {\"ACCEPTED\" if is_ok else \"REJECTED\"} - {reason}')
"
```

**Expected:**
- dark_frame: REJECTED (too_dark)
- bright_frame: REJECTED (overexposed)
- solid_color: REJECTED (solid_color or low_contrast)

### Step 6: Full Pipeline Test (if dependencies fixed)
```bash
python test_pipeline.py --video "data/videos/Sneaky Thieves Caught Stealing Phones On Camera - Newsflare (1080p, h264) (1).mp4" --max-frames 10
```

### Step 7: Dashboard Integration Test
Start the services and test via dashboard:
```bash
# Terminal 1: Start API
python -m src.api

# Terminal 2: Start Dashboard
streamlit run demo/dashboard.py

# In browser:
# 1. Upload the video
# 2. Monitor processing status
# 3. Check extracted frames
# 4. Review alerts
# 5. Verify CRITICAL alerts appear
```

## Evaluation Criteria

### PASS Criteria:
1. Frame extraction works with quality filtering
2. Vision analysis produces threat assessments
3. People counting is accurate (within ±1 of actual)
4. Retail theft scenarios trigger CRITICAL/HIGH alerts
5. Text/menu frames are rejected during extraction
6. No IndexError or import errors

### FAIL Criteria:
1. pymongo import error (must be fixed)
2. People count is 1 when there are 5+ people
3. Threat level is CLEAR for theft scenarios
4. Text overlay frames are analyzed
5. Pipeline crashes with IndexError

## Report Format
After testing, provide:

1. **Frame Extraction Results**:
   - Frames extracted: X
   - Frames rejected: Y (reasons)
   - Quality filtering: PASS/FAIL

2. **Vision Analysis Results**:
   - Threat level detected: [actual]
   - Expected threat level: [expected]
   - People count: [actual]
   - Expected people count: [expected]
   - Description quality: [PASS/FAIL]

3. **Overall Assessment**:
   - Pipeline status: WORKING/BROKEN
   - Threat detection accuracy: [percentage]
   - Critical issues: [list]

4. **Recommendations**:
   - What needs to be fixed
   - What needs to be improved
   - What is working well
