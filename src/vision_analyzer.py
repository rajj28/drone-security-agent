"""
vision_analyzer.py — Analyzes each frame using GPT-4o Vision and generates structured security assessment JSON.

- Loads JPG image, encodes to base64
- Sends to GPT-4o Vision with security prompt and telemetry context
- Parses and saves analysis JSON per frame
- OPTIMIZED: Supports multi-model analysis (CLIP, BLIP, GPT-4o) for enhanced accuracy
"""

import sys

# Force UTF-8 on stdout/stderr to prevent UnicodeEncodeError with emoji/unicode
# on Windows (cp1252) when running as a subprocess or under uvicorn.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import base64
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Union
from PIL import Image
from src.config import settings
from src.gemini_client import generate_vision
from src.api import get_latest_extracted_folder
import os

from src.session_bootstrap import apply_session_layout

apply_session_layout()

SESSION_CONTEXT_PATH = settings.SESSION_DIR / "session_context.json"
CONTEXT_SUMMARIES_PATH = settings.SESSION_DIR / "context_summaries.json"

# Configuration for analyzer selection (read from centralized settings)
USE_ULTIMATE_ANALYZER = bool(getattr(settings, "USE_ULTIMATE_ANALYZER", False))
USE_BLIP_ANALYZER = bool(getattr(settings, "USE_BLIP_ANALYZER", False))
USE_CLIP_ANALYZER = bool(getattr(settings, "USE_CLIP_ANALYZER", False))
USE_CLOUD_ANALYZER = bool(settings.USE_CLOUD_ANALYZER)


def cloud_analyzer_enabled() -> bool:
    """Runtime check for the cloud enhancer (HF CLIP+BLIP cross-check).

    The pipeline sets USE_CLOUD_ANALYZER in os.environ per run (driven by the
    dashboard's "Enable Cloud Enhancers" toggle), so the env value must win
    over the boot-time settings default.
    """
    env = os.environ.get("USE_CLOUD_ANALYZER")
    if env is not None:
        return env.lower() in ("1", "true", "yes")
    return USE_CLOUD_ANALYZER

print(f"Analyzer Configuration:")
print(f"  Ultimate Analyzer (Local CLIP+BLIP+GPT-4o): {USE_ULTIMATE_ANALYZER}")
print(f"  Cloud Analyzer (HF CLIP+BLIP + Gemini): {USE_CLOUD_ANALYZER}")
print(f"  BLIP Analyzer: {USE_BLIP_ANALYZER}")
print(f"  CLIP Analyzer: {USE_CLIP_ANALYZER}")
print(f"  Standard Gemini Vision: {not (USE_ULTIMATE_ANALYZER or USE_BLIP_ANALYZER or USE_CLIP_ANALYZER or USE_CLOUD_ANALYZER)}")

# System and user prompts - Super AI Drone Surveillance (Universal)
SYSTEM_PROMPT = """
You are an elite AI drone surveillance security analyst. You can analyze ANY environment — retail, industrial, residential, traffic, public spaces, campuses, construction sites, offices, warehouses, events, and more. Your expertise is SITUATION UNDERSTANDING: you first IDENTIFY the environment, then apply context-appropriate security rules.

=== STEP 1: AUTO-DETECT ENVIRONMENT ===
Before analyzing threats, determine the scene type from visual cues:
- RETAIL/COMMERCIAL: Display cases, merchandise, counters, POS systems, shopping aisles
- INDUSTRIAL/WAREHOUSE: Machinery, pallets, forklifts, loading docks, storage racks
- RESIDENTIAL: Houses, apartments, driveways, yards, fences, mailboxes
- PARKING/TRAFFIC: Vehicles, lanes, parking spots, traffic signs, roads
- CAMPUS/OFFICE: Buildings, walkways, entrances, lobbies, corridors
- CONSTRUCTION: Equipment, scaffolding, materials, barriers, hard hat areas
- PUBLIC SPACE: Parks, plazas, sidewalks, bus stops, train stations
- PERIMETER/GATE: Fences, walls, entry points, guard booths, barriers
- OTHER: Determine from visual context — always specify what you see

=== STEP 2: CORE ANALYSIS PROCESS ===
1. OBSERVE: What exactly is visible? Count people precisely. Note exact positions, objects, vehicles.
2. IDENTIFY ENVIRONMENT: What type of area is this? What are the key features?
3. CONTEXTUALIZE: Is this a restricted zone? After hours? High-value area? Active work zone?
4. CORRELATE: Does each person's behavior match what is EXPECTED in this environment?
5. THREAT ASSESS: Apply the environment-specific threat rules below.

=== UNIVERSAL THREAT LEVELS ===
- CRITICAL: Immediate danger — ANY environment
  * Weapon visible, fire, explosion, physical assault, medical emergency
  * Forced entry, lock picking, window breaking, fence cutting
  * After-hours intrusion into any secured area
  * Active theft in progress (grabbing items and fleeing)
  * Vehicle ramming barriers, aggressive driving toward people
  * Person down / collapsed (potential injury)

- HIGH: Active security concern requiring immediate attention
  * Trespassing in non-public, restricted, or staff-only areas
  * Person tampering with equipment, locks, security cameras, or infrastructure
  * Concealing items under clothing or in bags (any environment)
  * Coordinated suspicious behavior (lookout + actor pattern)
  * Vehicle following people, circling area repeatedly, or blocking exits
  * Person climbing fences, walls, or accessing rooftops
  * Unauthorized person handling valuable items, equipment, or inventory
  * Person fleeing / running from a scene while carrying objects
  * Vandalism or property damage in progress

- MEDIUM: Suspicious but unconfirmed — warrants monitoring
  * Loitering without clear purpose for extended time
  * Unattended bags, packages, or objects in public areas
  * Vehicle parked in no-parking zones, fire lanes, or unusual spots
  * Person acting nervous, repeatedly checking surroundings
  * Person photographing security infrastructure (cameras, locks, access points)
  * Unusual gathering or crowd forming without apparent reason
  * Person trying door handles, checking windows, or testing access points
  * Unfamiliar person in employee/staff areas during business hours

- LOW: Minor concern
  * Unfamiliar person in public area behaving normally
  * Minor rule violations (jaywalking, parking slightly wrong)
  * Person lingering briefly but with apparent purpose

- CLEAR: Normal activity appropriate for the environment
  * Workers performing expected duties in work areas
  * Pedestrians walking through public spaces normally
  * Vehicles moving through traffic or parked in designated spots
  * Deliveries, maintenance, or service activities with proper access
  * Normal social activity (people talking, eating, exercising)

=== ENVIRONMENT-SPECIFIC THREAT INDICATORS ===

RETAIL / COMMERCIAL:
- Person behind counter without uniform/badge = HIGH (unauthorized access)
- Reaching into display cases, shelves, or registers = HIGH (potential theft)
- Concealing merchandise under clothing = HIGH (shoplifting)
- Multiple people clustered while one handles items (distraction tactic) = HIGH
- Grabbing multiple items rapidly = HIGH (sweeping behavior)
- Quick hand movements near merchandise + checking surroundings = MEDIUM+

INDUSTRIAL / WAREHOUSE:
- Unauthorized entry into storage or machinery areas = HIGH
- Person near hazardous materials without PPE = MEDIUM
- Tampering with equipment or safety systems = CRITICAL
- Loading/unloading without proper identification = MEDIUM
- Person photographing proprietary equipment = MEDIUM

RESIDENTIAL:
- Person checking windows/doors of unoccupied homes = HIGH
- Unfamiliar person in backyard or private property = HIGH
- Vehicle idling near residences, especially at night = MEDIUM
- Person carrying items FROM a residence without clear reason = HIGH
- Package theft from porches/doorsteps = HIGH

PARKING / TRAFFIC:
- Vehicle circling the same area repeatedly = MEDIUM
- Person checking car doors or looking into vehicles = HIGH
- Vehicle blocking emergency lanes or exits = MEDIUM
- Person loitering in parking structure without vehicle = MEDIUM
- Aggressive driving or reckless vehicle behavior = HIGH

CAMPUS / OFFICE:
- After-hours access without credentials = HIGH
- Tailgating through secure doors = MEDIUM
- Person in server rooms or restricted floors = HIGH
- Unattended item in high-traffic area = MEDIUM

CONSTRUCTION SITES:
- Unauthorized entry outside work hours = HIGH
- Person near heavy equipment without authorization = MEDIUM
- Theft of materials or tools = HIGH
- Missing safety barriers or unsafe conditions = MEDIUM

PUBLIC SPACES / EVENTS:
- Unattended bags in crowded areas = MEDIUM
- Person moving against crowd flow or acting erratically = MEDIUM
- Altercation or aggressive behavior between people = HIGH
- Person filming/photographing restricted infrastructure = LOW

=== BEHAVIORAL CORRELATION (UNIVERSAL) ===
Combine multiple signals for escalation:
- Restricted area + After hours + ANY person = HIGH minimum
- Concealment behavior + Near valuables + Checking surroundings = HIGH
- Running + Carrying items + Away from building/area = HIGH (theft in progress)
- Multiple suspicious people + Coordinated movement = HIGH (organized threat)
- Tampering + Infrastructure + Alone = CRITICAL (sabotage risk)
- Fence/wall climbing + Restricted area = CRITICAL (intrusion)

=== CRITICAL RULE: DO NOT ASSUME ROLES ===
NEVER assume someone is an "employee", "guard", "customer", "resident", or "worker" unless
they wear a VISIBLE uniform, badge, or safety gear. Report OBSERVED ACTIONS, not assumed roles.

Respond ONLY with valid JSON matching the exact format specified.
"""
USER_PROMPT_TEMPLATE = """
Analyze this drone surveillance frame for security threats. First IDENTIFY the environment type, then apply context-appropriate security analysis.

=== CONTEXT ===
- Location: {location}
- Time: {timestamp}
- Drone altitude: {altitude}m
- Zone Type: {zone_type}
- After Hours: {is_after_hours}

=== ANALYSIS INSTRUCTIONS ===
1. IDENTIFY ENVIRONMENT: What type of area is this? (retail, industrial, residential, parking, campus, construction, public space, etc.)
2. COUNT PRECISELY: Count every visible person in the frame (exact number, not approximations)
3. DESCRIBE EACH PERSON: Position, body posture, hand activity, movement direction, what they are interacting with
4. NOTE ALL VEHICLES: Type, color, position, movement, any plates visible
5. IDENTIFY OBJECTS: Equipment, bags, merchandise, tools, weapons, packages — anything security-relevant
6. ASSESS BEHAVIOR: Does each person's behavior match what is EXPECTED in this environment?
7. DETECT ANOMALIES: Anything out of place — wrong time, wrong area, wrong behavior for the context

=== DO NOT ASSUME ROLES (CRITICAL) ===
Do NOT assume someone is an "employee", "guard", "customer", "resident", or "delivery person" unless
they wear a VISIBLE uniform, badge, vest, or other identifying markers. Report ONLY observed facts.
"Person behind counter handling items" is a FACT. "Employee serving customer" is an ASSUMPTION — avoid it.

=== UNIVERSAL THREAT DETECTION ===
For EACH person visible, analyze:
- WHERE are they positioned relative to restricted/sensitive areas?
- WHAT are their hands doing? (carrying, reaching, concealing, tampering, idle)
- HOW are they moving? (purposeful, loitering, evasive, hurried, sneaking)
- WHO are they interacting with and how? (cooperative, aggressive, avoiding contact)
- Are they aware of surveillance? (looking at cameras, checking surroundings repeatedly)

Flag as suspicious (MEDIUM or higher) if you observe:
- Hands interacting with items/equipment/property they shouldn't be accessing
- Concealment behavior (hiding items under clothing, in bags, behind objects)
- Evasive movement patterns (avoiding cameras, staying in shadows, checking exits)
- Coordinated behavior between multiple people (lookout + actor pattern)
- Tampering with locks, security devices, infrastructure, or equipment
- Presence in restricted areas without visible authorization
- Carrying items away from where they belong

=== KEY QUESTIONS TO ANSWER ===
- What type of environment/scene is this?
- How many people are ACTUALLY visible? (precise count)
- Where exactly is each person positioned and what are they doing?
- What are each person's hands specifically doing?
- Are there any vehicles, and what are they doing?
- Is anything suspicious, unusual, or out of place for THIS type of environment?
- Does anyone appear to be where they shouldn't be or doing something they shouldn't?

Respond ONLY with valid JSON:
{{
  "threat_level": "CRITICAL|HIGH|MEDIUM|LOW|CLEAR",
  "threat_type": "loitering|trespassing|unauthorized_vehicle|suspicious_behavior|theft_behavior|confrontation|vandalism|intrusion|tampering|safety_violation|clear",
  "vlm_description": "detailed scene description - identify environment type, describe all people, vehicles, objects, and activities with specific positions and actions",
  "scene_type": "parking_lot|warehouse|retail|perimeter|gate|garage|interior|exterior|residential|campus|construction|public_space|traffic|industrial|office|unknown",
  "people_count": 0,
  "objects_detected": ["person", "vehicle", "bag", "equipment"],
  "person_features": [
    "Person 1: [position in frame + body posture + hand activity + movement + what they are doing + any identifying features]",
    "Person 2: [same format]"
  ],
  "vehicles_detected": ["white sedan", "delivery truck"],
  "activity": "precise description of what people/vehicles are doing in this environment",
  "security_signals": ["unauthorized_area_access", "concealment_behavior", "evasive_movement", "tampering", "loitering"],
  "suspicious_elements": ["person_in_restricted_area", "unusual_behavior_for_environment", "unattended_object"],
  "reasoning": "EXPLAIN: This appears to be a [environment type]. Person is [doing X] at [location] which is [threat level] because [specific contextual reason]. Connect what you see to WHY it matters in THIS environment.",
  "recommended_action": "specific security response appropriate for this environment and threat level",
  "confidence": 0.85
}}

IMPORTANT: In your reasoning, first state what environment type you identified, then explain WHY the observed behavior is or isn't concerning in THAT specific context. Connect the dots between behavior, location, and threat level.
"""


def _as_string_list(value: Any) -> list:
    """Normalizes common list-like values into a list of strings."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _extract_partial_json(json_str: str) -> Dict[str, Any]:
    """Extract partial data from truncated/incomplete JSON string."""
    result = {}
    
    # vlm_description — handle truncated JSON (unclosed string)
    vlm_match = re.search(r'"vlm_description"\s*:\s*"(.*)', json_str, re.DOTALL)
    if vlm_match:
        desc = vlm_match.group(1)
        if '",' in desc:
            desc = desc.split('",', 1)[0]
        desc = desc.replace('\\n', '\n').replace('\\"', '"').strip()
        if len(desc) > 15:
            result["vlm_description"] = desc
    # threat_level from partial JSON
    threat_match = re.search(r'"threat_level"\s*:\s*"([^"]+)"', json_str)
    if threat_match:
        result["threat_level"] = threat_match.group(1)
    
    # Try to extract scene_type
    scene_match = re.search(r'"scene_type"\s*:\s*"([^"]*)"', json_str)
    if scene_match:
        result["scene_type"] = scene_match.group(1)
    
    # Try to extract people_count
    people_match = re.search(r'"people_count"\s*:\s*(\d+)', json_str)
    if people_match:
        result["people_count"] = int(people_match.group(1))
    
    # Try to extract objects_detected array
    objects_match = re.search(r'"objects_detected"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
    if objects_match:
        try:
            objects_str = objects_match.group(1)
            # Extract quoted strings
            objects = re.findall(r'"([^"]*)"', objects_str)
            if objects:
                result["objects_detected"] = objects
        except:
            pass
    
    # Try to extract object_details array - handle partial array
    obj_details_match = re.search(r'"object_details"\s*:\s*(\[.*?\])(?:,\s*"|$)', json_str, re.DOTALL)
    if obj_details_match:
        try:
            details_str = obj_details_match.group(1)
            # Try to parse as JSON, if fails, try to fix common issues
            try:
                result["object_details"] = json.loads(details_str)
            except:
                # Extract individual objects with regex as fallback
                pass
        except:
            pass
    
    # Try to extract person_features array - handle partial/truncated
    # Look for opening bracket and capture until next top-level field or end
    person_match = re.search(r'"person_features"\s*:\s*(\[.*?)(?:,\s*"[a-z_]+"\s*:|\}|$)', json_str, re.DOTALL)
    if person_match:
        try:
            person_str = person_match.group(1).strip()
            # Try to complete truncated JSON by adding missing brackets
            if not person_str.endswith(']'):
                # Count opening brackets and add closing ones
                open_count = person_str.count('[') + person_str.count('{')
                close_count = person_str.count(']') + person_str.count('}')
                person_str += ']' * (open_count - close_count)
            try:
                result["person_features"] = json.loads(person_str)
            except:
                # Try extracting individual person objects as fallback
                person_objects = re.findall(r'\{\s*"id"\s*:\s*"([^"]+)".*?\}', person_str, re.DOTALL)
                if person_objects:
                    # Build minimal person features from IDs
                    result["person_features"] = [{"id": pid} for pid in person_objects]
        except Exception as e:
            print(f"Failed to extract person_features: {e}")
    
    # Try to extract activity
    activity_match = re.search(r'"activity"\s*:\s*"([^"]*)"', json_str)
    if activity_match:
        result["activity"] = activity_match.group(1)
    
    # Try to extract recommended_action
    action_match = re.search(r'"recommended_action"\s*:\s*"([^"]*)"', json_str)
    if action_match:
        result["recommended_action"] = action_match.group(1)
    
    return result

def _normalize_analysis(analysis: Dict[str, Any], telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures the analysis payload has the downstream fields needed for reasoning.
    
    FIXED: Now properly extracts fields from raw_response if initial parsing failed,
    including handling truncated/incomplete JSON responses.
    """
    normalized = dict(analysis)
    
    # If parsing failed, try to extract from raw_response
    if normalized.get("_parsing_failed") and normalized.get("raw_response"):
        raw = normalized["raw_response"]
        parsed = None
        
        # First try: Extract JSON from markdown fences
        try:
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                json_str = raw.split("```")[1].strip()
            else:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1:
                    json_str = raw[start:end+1]
                else:
                    json_str = raw
            
            # Try full JSON parse first
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                # Second try: Extract partial data from truncated JSON
                print(f"JSON truncated, attempting partial extraction...")
                parsed = _extract_partial_json(json_str)
                
        except Exception as e:
            print(f"Failed to extract from raw_response: {e}")
        
        # Merge parsed data with normalized (only for empty/missing fields)
        if parsed:
            for key, value in parsed.items():
                if key not in normalized or not normalized[key] or normalized[key] == "unknown" or normalized[key] == "Scene analysis unavailable.":
                    normalized[key] = value
            # Remove _parsing_failed flag since we successfully extracted data
            if any(k in parsed for k in ["vlm_description", "scene_type", "people_count"]):
                normalized.pop("_parsing_failed", None)
    
    # Set defaults for missing fields
    normalized.setdefault("frame_id", telemetry.get("frame_id"))
    normalized.setdefault("scene_type", "unknown")
    normalized.setdefault("object_details", [])
    normalized.setdefault("person_features", [])
    normalized.setdefault("vehicle_details", [])
    normalized.setdefault("security_signals", [])
    normalized.setdefault("alert_reasoning", normalized.get("recommended_action", ""))
    normalized.setdefault("alert_priority_signals", [])
    
    # Extract security signals from person_features actions
    person_features = normalized.get("person_features", [])
    suspicious_actions = ["reaching", "concealing", "hiding", "grabbing", "palming", "snatching", "pocketing", "loitering", "lurking", "sneaking"]
    security_signals = []
    suspicious_elements = []
    
    for idx, person in enumerate(person_features):
        if isinstance(person, dict):
            actions = person.get("actions", [])
            person_id = person.get("id", f"person_{idx+1}")
            
            for action in actions:
                action_lower = action.lower() if isinstance(action, str) else ""
                # Check for suspicious actions
                for suspicious in suspicious_actions:
                    if suspicious in action_lower:
                        signal = f"{person_id}: {action}"
                        if signal not in security_signals:
                            security_signals.append(signal)
                        if action not in suspicious_elements:
                            suspicious_elements.append(action)
        elif isinstance(person, str):
            # Parse string descriptions for suspicious actions
            person_lower = person.lower()
            person_id = f"person_{idx+1}"
            
            for suspicious in suspicious_actions:
                if suspicious in person_lower:
                    # Extract the relevant part of the sentence containing the suspicious action
                    words = person.split()
                    for i, word in enumerate(words):
                        if suspicious in word.lower():
                            # Get context around the suspicious word (3 words before and after)
                            start = max(0, i - 3)
                            end = min(len(words), i + 4)
                            context = ' '.join(words[start:end])
                            signal = f"{person_id}: {context}"
                            if signal not in security_signals:
                                security_signals.append(signal)
                            if suspicious not in suspicious_elements:
                                suspicious_elements.append(suspicious)
                           # Scan activity and vlm_description for suspicious actions to ensure coverage
    activity_lower = normalized.get("activity", "").lower()
    description_lower = normalized.get("vlm_description", "").lower()
    
    for suspicious in suspicious_actions + ["stealing", "theft", "shoplifting", "robbery"]:
        if suspicious in activity_lower or suspicious in description_lower:
            sig = f"scene: detected potential {suspicious} activity"
            if sig not in security_signals:
                security_signals.append(sig)
            if suspicious not in suspicious_elements:
                suspicious_elements.append(suspicious)

    # Merge extracted signals with existing ones
    if security_signals:
        existing_signals = normalized.get("security_signals", []) or []
        for sig in security_signals:
            if sig not in existing_signals:
                existing_signals.append(sig)
        normalized["security_signals"] = existing_signals
    
    if suspicious_elements:
        existing_suspicious = normalized.get("suspicious_elements", []) or []
        for elem in suspicious_elements:
            if elem not in existing_suspicious:
                existing_suspicious.append(elem)
        normalized["suspicious_elements"] = existing_suspicious
    
    # Update alert reasoning if security signals detected
    current_reasoning = normalized.get("alert_reasoning", "")
    benign_messages = ["Review frame manually", "No immediate action required", "", "N/A"]
    
    if security_signals:
        if any(msg in current_reasoning for msg in benign_messages) or not current_reasoning:
            normalized["alert_reasoning"] = f"Security threat detected: {', '.join(security_signals)}. Review frame immediately for suspicious activity."
    
    # RULE-BASED ALERT LAYER: Adjust threat level based on telemetry context
    threat_level = normalized.get("threat_level", "UNKNOWN")
    threat_type = normalized.get("threat_type", "unknown")
    is_after_hours = telemetry.get("is_after_hours", False)
    location = telemetry.get("location", "unknown")
    timestamp = telemetry.get("timestamp", "")
    
    # Rule 0: Convert UNKNOWN to CLEAR or MEDIUM based on security signals
    if threat_level == "UNKNOWN":
        if security_signals:
            threat_level = "MEDIUM"
            threat_type = "suspicious_behavior"
        else:
            threat_level = "CLEAR"
            threat_type = "clear"
            
    # Rule 0.5: If security signals are present, upgrade CLEAR or LOW threat levels to at least MEDIUM (or HIGH for theft behavior)
    if security_signals and threat_level in ["CLEAR", "LOW", "UNKNOWN"]:
        has_theft_action = any(any(theft in sig.lower() for theft in ["pocket", "conceal", "hide", "grab", "steal", "theft", "shoplift", "reach"]) for sig in security_signals)
        if has_theft_action:
            threat_level = "HIGH"
            threat_type = "theft_behavior"
        else:
            threat_level = "MEDIUM"
            threat_type = "suspicious_behavior"
    
    # Rule 1: Escalate to HIGH if after hours and any threat detected
    if is_after_hours and threat_level in ["MEDIUM", "LOW", "UNKNOWN"]:
        threat_level = "HIGH"
        threat_type = f"after_hours_{threat_type}" if threat_type != "clear" else "after_hours_activity"
        normalized["alert_reasoning"] = f"Threat escalated to HIGH due to after-hours activity at {location}. {normalized.get('alert_reasoning', '')}"
    
    # Rule 2: CRITICAL for weapons, fire, forced entry (regardless of time)
    critical_signals = ["weapon", "fire", "assault", "forced entry", "break-in"]
    if any(sig in str(security_signals).lower() for sig in critical_signals):
        threat_level = "CRITICAL"
        normalized["alert_reasoning"] = f"CRITICAL threat detected: Immediate danger at {location}. {normalized.get('alert_reasoning', '')}"
    
    # Rule 3: Vehicle in restricted zone = MEDIUM minimum
    if location in ["restricted_zone", "warehouse", "perimeter"]:
        vehicles = normalized.get("vehicles_detected", [])
        if vehicles and threat_level == "CLEAR":
            threat_level = "MEDIUM"
            threat_type = "unauthorized_vehicle"
            normalized["alert_reasoning"] = f"Unauthorized vehicle detected in {location}. Review required."
    
    # Rule 4: Loitering across multiple frames = escalate
    if "loitering" in str(security_signals).lower() and threat_level == "MEDIUM":
        threat_level = "HIGH"
        normalized["alert_reasoning"] = f"Persistent loitering detected at {location}. Escalated to HIGH."
    
    # Update normalized values
    normalized["threat_level"] = threat_level
    normalized["threat_type"] = threat_type
    normalized["threat_assessment"] = threat_level
 
     # Ensure confidence is a float between 0 and 1
    raw_conf = normalized.get("confidence")
    # Handle case where confidence is a nested dict (e.g. {"confidence": 0.9})
    if isinstance(raw_conf, dict):
        conf = raw_conf.get("confidence")
    else:
        conf = raw_conf
    if conf is None:
        normalized["confidence"] = 0.85
    else:
        try:
            normalized["confidence"] = float(conf)
            if not (0.0 <= normalized["confidence"] <= 1.0):
                normalized["confidence"] = 0.85
        except Exception:
            normalized["confidence"] = 0.85

    normalized["objects_detected"] = _as_string_list(normalized.get("objects_detected", []))
    normalized["vehicles_detected"] = _as_string_list(normalized.get("vehicles_detected", []))
    normalized["suspicious_elements"] = _as_string_list(normalized.get("suspicious_elements", []))
    normalized["security_signals"] = _as_string_list(normalized.get("security_signals", []))
    normalized["alert_priority_signals"] = _as_string_list(normalized.get("alert_priority_signals", []))

    try:
        normalized["people_count"] = int(normalized.get("people_count", 0) or 0)
    except Exception:
        normalized["people_count"] = 0

    if not normalized.get("vlm_description"):
        normalized["vlm_description"] = "Scene analysis unavailable."
    if not normalized.get("activity"):
        normalized["activity"] = "Unknown activity"
    if not normalized.get("recommended_action"):
        normalized["recommended_action"] = "Review frame manually"

    return normalized

def image_to_base64(image_path: Path) -> str:
    """Converts an image to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def extract_json_payload(content: str) -> Dict[str, Any]:
    """Extracts the first JSON object from a model response.

    The model may wrap JSON in markdown fences or add brief prose around it,
    so this helper strips common wrappers and then falls back to balanced-brace
    extraction before parsing.
    
    FIXED: Now properly extracts all fields from raw_response if initial parse fails.
    """
    cleaned = content.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        candidate = candidate.replace("\r\n", "\n")
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # If all parsing fails, return raw_response for later extraction
    return {"raw_response": content, "_parsing_failed": True}


def load_session_context() -> Dict[str, Any]:
    """Loads or initializes the rolling session context state."""
    default_context = {
        "frames_analyzed": 0,
        "total_alerts": 0,
        "high_alerts": 0,
        "medium_alerts": 0,
        "people_detected": 0,
        "vehicles_detected": 0,
        "locations_visited": [],
        "incidents": [],
        "running_narrative": "Session started.",
    }

    if SESSION_CONTEXT_PATH.exists():
        with open(SESSION_CONTEXT_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return {**default_context, **data}

    return default_context


def load_context_summaries() -> Dict[str, Any]:
    """Loads or initializes the per-frame context summaries container."""
    if CONTEXT_SUMMARIES_PATH.exists():
        with open(CONTEXT_SUMMARIES_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data.setdefault("session_id", f"SESSION_{time.strftime('%Y%m%d_%H%M%S')}")
            data.setdefault("summaries", [])
            return data

    return {
        "session_id": f"SESSION_{time.strftime('%Y%m%d_%H%M%S')}",
        "summaries": [],
    }


def derive_alert_from_analysis(analysis: Dict[str, Any], telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """Derives a lightweight alert summary from a frame analysis."""
    alert_triggered = False
    severity = "NONE"
    alert_type = None
    reasons = []

    objects_detected = analysis.get("objects_detected", []) or []
    people_count = int(analysis.get("people_count", 0) or 0)
    threat_assessment = str(analysis.get("threat_level") or analysis.get("threat_assessment", "none") or "none").lower()
    activity = str(analysis.get("activity", "") or "").lower()
    security_signals = [str(item).lower() for item in analysis.get("security_signals", []) or []]
    alert_priority_signals = [str(item).lower() for item in analysis.get("alert_priority_signals", []) or []]
    vehicle_details = analysis.get("vehicle_details", []) or []
    person_features = analysis.get("person_features", []) or []

    if telemetry.get("is_after_hours") and "person" in objects_detected:
        alert_triggered = True
        severity = "HIGH"
        alert_type = "after_hours_person"
        reasons.append("person detected after hours")
    elif telemetry.get("is_restricted_zone") and "person" in objects_detected:
        alert_triggered = True
        severity = "HIGH"
        alert_type = "restricted_zone_person"
        reasons.append("person detected in restricted zone")
    elif people_count > 3:
        alert_triggered = True
        severity = "MEDIUM"
        alert_type = "crowd_detected"
        reasons.append("crowd detected")
    elif "loiter" in activity:
        alert_triggered = True
        severity = "MEDIUM"
        alert_type = "loitering_detected"
        reasons.append("loitering activity observed")
    elif threat_assessment not in ["clear", "low", "none", "unknown"]:
        alert_triggered = True
        severity = "HIGH" if any(t in threat_assessment for t in ["high", "critical"]) else "MEDIUM"
        alert_type = "threat_assessment_high"
        reasons.append(f"model threat assessment is {threat_assessment}")

    if not alert_triggered:
        if any(signal in {"after_hours_presence", "restricted_zone_presence", "restricted_zone_vehicle"} for signal in security_signals):
            alert_triggered = True
            severity = "MEDIUM"
            alert_type = "security_signal_detected"
            reasons.append("security signal raised by the vision model")
        elif any(signal in {"person_after_hours", "vehicle_loitering", "unknown_vehicle", "masked_person"} for signal in alert_priority_signals):
            alert_triggered = True
            severity = "MEDIUM"
            alert_type = "priority_signal_detected"
            reasons.append("priority security signal detected")

    if not reasons and person_features:
        suspicious_traits = []
        for person in person_features:
            if not isinstance(person, dict):
                continue
            appearance = [str(item).lower() for item in person.get("appearance", []) or []]
            actions = [str(item).lower() for item in person.get("actions", []) or []]
            if any(term in appearance for term in ["hood", "mask", "covered_face", "dark_clothes"]):
                suspicious_traits.append("person appearance is security-relevant")
            if any(term in actions for term in ["loitering", "watching", "following", "running"]):
                suspicious_traits.append("person behavior is suspicious")
        if suspicious_traits and not alert_triggered:
            alert_triggered = True
            severity = "MEDIUM"
            alert_type = "person_behavior_signal"
            reasons.extend(suspicious_traits)

    if not alert_triggered and vehicle_details:
        vehicle_signals = []
        for vehicle in vehicle_details:
            if not isinstance(vehicle, dict):
                continue
            color = str(vehicle.get("color", "")).lower()
            vehicle_type = str(vehicle.get("type", "")).lower()
            position = str(vehicle.get("position", "")).lower()
            plate_visible = bool(vehicle.get("plate_visible", False))
            if color in {"black", "white", "silver", "gray", "grey", "blue", "red"}:
                vehicle_signals.append(f"vehicle color observed: {color}")
            if position in {"parked", "stopped", "idling"} and telemetry.get("is_after_hours"):
                vehicle_signals.append("vehicle stationary after hours")
            if not plate_visible and telemetry.get("is_restricted_zone"):
                vehicle_signals.append("plate not visible in restricted zone")
            if vehicle_type in {"truck", "van", "suv", "sedan"}:
                vehicle_signals.append(f"vehicle type: {vehicle_type}")
        if vehicle_signals:
            reasons.extend(vehicle_signals[:3])
            if telemetry.get("is_after_hours") or telemetry.get("is_restricted_zone"):
                alert_triggered = True
                severity = "MEDIUM"
                alert_type = "vehicle_context_signal"

    return {
        "alert_triggered": alert_triggered,
        "severity": severity,
        "alert_type": alert_type,
        "reasoning": "; ".join(reasons) if reasons else "No alert criteria matched.",
        "reasoning_signals": reasons,
    }


def build_context_summary(
    frame_id: str,
    telemetry: Dict[str, Any],
    analysis: Dict[str, Any],
    session_context: Dict[str, Any],
    alert_summary: Dict[str, Any],
) -> str:
    """Builds a concise cumulative summary for the current frame."""
    frames_analyzed = session_context.get("frames_analyzed", 0)
    total_alerts = session_context.get("total_alerts", 0)
    people_detected = session_context.get("people_detected", 0)
    vehicles_detected = session_context.get("vehicles_detected", 0)
    location = telemetry.get("location", "unknown location")
    timestamp = telemetry.get("timestamp", "unknown time")
    activity = analysis.get("activity", "No activity reported")
    threat = str(analysis.get("threat_assessment", "none") or "none").lower()
    reasoning = str(analysis.get("alert_reasoning", "") or "").strip()
    priority_signals = ", ".join(analysis.get("alert_priority_signals", []) or [])

    summary_bits = [
        f"Frame {frame_id} analyzed at {location} ({timestamp}).",
        f"Cumulative status: {frames_analyzed} frames, {total_alerts} alerts, {people_detected} people, {vehicles_detected} vehicles.",
        f"Current frame activity: {activity}.",
    ]

    if alert_summary.get("alert_triggered"):
        summary_bits.append(
            f"Alert: {alert_summary.get('severity', 'UNKNOWN')} {alert_summary.get('alert_type', 'unknown_alert')} triggered."
        )
        if alert_summary.get("reasoning"):
            summary_bits.append(f"Alert reasoning: {alert_summary.get('reasoning')}.")
    else:
        summary_bits.append("No alert triggered for this frame.")

    if threat != "none":
        summary_bits.append(f"Threat assessment for this frame: {threat}.")

    if reasoning:
        summary_bits.append(f"Vision reasoning: {reasoning}.")
    if priority_signals:
        summary_bits.append(f"Priority signals: {priority_signals}.")

    return " ".join(summary_bits)


def update_session_context(
    session_context: Dict[str, Any],
    telemetry: Dict[str, Any],
    analysis: Dict[str, Any],
    alert_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Updates rolling session metrics from the latest frame."""
    session_context["frames_analyzed"] = int(session_context.get("frames_analyzed", 0)) + 1

    session_context["people_detected"] = int(session_context.get("people_detected", 0)) + int(
        analysis.get("people_count", 0) or 0
    )

    session_context["vehicles_detected"] = int(session_context.get("vehicles_detected", 0)) + len(
        analysis.get("vehicles_detected", []) or []
    )

    location = telemetry.get("location")
    locations = session_context.setdefault("locations_visited", [])
    if location and location not in locations:
        locations.append(location)

    if alert_summary.get("alert_triggered"):
        session_context["total_alerts"] = int(session_context.get("total_alerts", 0)) + 1
        if alert_summary.get("severity") == "HIGH":
            session_context["high_alerts"] = int(session_context.get("high_alerts", 0)) + 1
        elif alert_summary.get("severity") == "MEDIUM":
            session_context["medium_alerts"] = int(session_context.get("medium_alerts", 0)) + 1

        incidents = session_context.setdefault("incidents", [])
        incidents.append(
            {
                "frame_id": telemetry.get("frame_id"),
                "timestamp": telemetry.get("timestamp"),
                "location": location,
                "severity": alert_summary.get("severity"),
                "alert_type": alert_summary.get("alert_type"),
                "reasoning": alert_summary.get("reasoning"),
                "threat_type": analysis.get("threat_type"),
                "description": analysis.get("vlm_description"),
                "activity": analysis.get("activity"),
            }
        )

    frames = int(session_context.get("frames_analyzed", 0))
    people = int(session_context.get("people_detected", 0))
    vehicles = int(session_context.get("vehicles_detected", 0))
    alerts = int(session_context.get("total_alerts", 0))
    locations_count = len(session_context.get("locations_visited", []))

    session_context["running_narrative"] = (
        f"Analyzed {frames} frames so far. {people} people detected, {vehicles} vehicles detected, "
        f"{alerts} alerts generated across {locations_count} locations."
    )

    return session_context


def persist_context_summary(
    frame_id: str,
    telemetry: Dict[str, Any],
    analysis: Dict[str, Any],
    session_context: Dict[str, Any],
    alert_summary: Dict[str, Any],
    context_store: Dict[str, Any],
) -> None:
    """Persists frame context via unified layer (JSON + structured + optional Mongo)."""
    from src.unified_context import get_unified_context

    ctx = get_unified_context(session_dir=settings.SESSION_DIR)
    ctx.record_frame(frame_id, telemetry, analysis, alert_summary)

def _use_offline_vision() -> bool:
    return os.getenv("OFFLINE_VISION", "").lower() in ("1", "true", "yes")


def _save_offline_result(
    frame_id: str,
    image_path: Path,
    telemetry: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    from src.offline_vision_fallback import analyze_frame_offline
    from src.api_retry import quota_exhausted

    reason = "OFFLINE_VISION=true" if _use_offline_vision() else "API quota/rate limit"
    print(f"[VISION] Using offline fallback ({reason}) for {frame_id}")
    result = analyze_frame_offline(frame_id, image_path, telemetry)
    out_path = output_dir / f"{frame_id}_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def analyze_frame(
    frame_id: str,
    image_path: Path,
    telemetry: Dict[str, Any],
    output_dir: Path = settings.ANALYSIS_DIR
) -> Dict[str, Any]:
    """
    Analyzes a frame using selected analyzer and saves the result as JSON.
    Returns the analysis dict.
    
    OPTIMIZED: Supports multiple analyzers based on configuration:
    - Ultimate: CLIP + BLIP + GPT-4o (most accurate)
    - BLIP: BLIP + GPT-4o (good balance)
    - CLIP: CLIP + GPT-4o (fast)
    - Standard: GPT-4o only (fastest)
    """
    print(f"\nAnalyzing {frame_id}...")

    from src.api_retry import quota_exhausted

    if _use_offline_vision() or quota_exhausted():
        return _save_offline_result(frame_id, image_path, telemetry, output_dir)
    
    # Use Ultimate Analyzer if configured
    if USE_ULTIMATE_ANALYZER:
        try:
            from src.ultimate_vision_analyzer import analyze_frame_ultimate
            print("Using Ultimate Analyzer (CLIP + BLIP + GPT-4o)...")
            result = analyze_frame_ultimate(image_path, telemetry, use_gpt4o=True)
            
            # Convert to standard format
            if result and result.get('gpt4o_analysis', {}).get('success'):
                gpt4o_content = result['gpt4o_analysis']['gpt4o_analysis']
                try:
                    analysis = extract_json_payload(gpt4o_content)
                except Exception:
                    analysis = {"raw_response": gpt4o_content}
                
                analysis = _normalize_analysis(analysis, telemetry)
                analysis["alert_reasoning"] = analysis.get("alert_reasoning") or analysis.get("recommended_action", "")
                analysis["reasoning_signals"] = analysis.get("reasoning_signals", [])
                
                # Add multi-model insights
                analysis["clip_threat_score"] = result['clip_analysis']['threat_score']
                analysis["blip_caption"] = result['blip_analysis'].get('blip_caption', '')
                analysis["blip_insights"] = result['blip_analysis'].get('blip_insights', {})
                analysis["overall_threat_level"] = result['overall_threat_level']
                
                final_result = {
                    "frame_id": frame_id,
                    "timestamp": telemetry["timestamp"],
                    "location": telemetry["location"],
                    **analysis,
                    "model_used": "ultimate-clip-blip-gpt4o",
                    "processing_time_ms": 0
                }
                
                out_path = output_dir / f"{frame_id}_analysis.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(final_result, f, indent=2)
                print(f"Analysis saved for {frame_id} (Ultimate Analyzer)")
                return final_result
        except Exception as e:
            print(f"Ultimate Analyzer failed, falling back to standard: {e}")
    
    # Use Cloud Analyzer if configured (Hugging Face API + Local GPT-4o)
    if cloud_analyzer_enabled():
        try:
            from src.cloud_enhanced_analyzer import analyze_frame_cloud
            print("[CLOUD] Using Cloud-Enhanced Analyzer (HF CLIP + BLIP + Gemini)...")
            print("   Requires HF_API_TOKEN environment variable")
            result = analyze_frame_cloud(image_path, telemetry)
            
            # Convert to standard format
            if result and result.get('gpt4o_analysis', {}).get('success'):
                gpt4o_content = result['gpt4o_analysis']['gpt4o_analysis']
                try:
                    analysis = extract_json_payload(gpt4o_content)
                except Exception:
                    analysis = {"raw_response": gpt4o_content}
                
                # Extract threat fields from cloud analyzer response
                gpt4o_result = result.get('gpt4o_analysis', {})
                if gpt4o_result.get('threat_level'):
                    analysis['threat_level'] = gpt4o_result['threat_level']
                if gpt4o_result.get('threat_type'):
                    analysis['threat_type'] = gpt4o_result['threat_type']
                
                analysis = _normalize_analysis(analysis, telemetry)
                analysis["alert_reasoning"] = analysis.get("alert_reasoning") or analysis.get("recommended_action", "")
                analysis["reasoning_signals"] = analysis.get("reasoning_signals", [])
                
                # Add cloud insights
                clip_data = result.get('clip_analysis', {}) or {}
                blip_data = result.get('blip_analysis', {}) or {}
                analysis["clip_threat_score"] = clip_data.get('threat_score', 0)
                analysis["blip_caption"] = blip_data.get('caption', '')
                analysis["blip_insights"] = blip_data.get('insights', {})
                analysis["overall_threat_level"] = result.get('overall_threat_level', 'UNKNOWN')
                
                final_result = {
                    "frame_id": frame_id,
                    "timestamp": telemetry["timestamp"],
                    "location": telemetry["location"],
                    **analysis,
                    "model_used": "cloud-clip-blip-gemini",
                    "processing_time_ms": result.get('processing_time_ms', 0),
                    "image_quality": result.get("image_quality"),
                }
                
                out_path = output_dir / f"{frame_id}_analysis.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(final_result, f, indent=2)
                print(f"[CLOUD] Cloud analysis saved for {frame_id}")
                return final_result
            else:
                print(f"[WARNING] Cloud Gemini analysis failed: {result.get('gpt4o_analysis', {}).get('error', 'Unknown')}")
        except Exception as e:
            print(f"[CLOUD] Cloud Analyzer failed, falling back to standard: {e}")

        if quota_exhausted():
            return _save_offline_result(frame_id, image_path, telemetry, output_dir)
    
    # Use BLIP Analyzer if configured
    if USE_BLIP_ANALYZER:
        try:
            from src.blip_vision_analyzer import analyze_with_blip
            print("Using BLIP Analyzer...")
            result = analyze_with_blip(image_path, telemetry)
            
            # Convert to standard format
            analysis = {
                "vlm_description": result.get('blip_caption', ''),
                "scene_type": result.get('blip_insights', {}).get('location', 'unknown'),
                "objects_detected": result.get('blip_insights', {}).get('main_objects', []),
                "people_count": result.get('blip_insights', {}).get('people_count', 0),
                "activity": result.get('blip_vqa_answers', {}).get('What are the people doing in this image?', ''),
                "threat_assessment": result.get('blip_threat_level', 'low').lower(),
                "suspicious_elements": [],
                "security_signals": []
            }
            
            analysis = _normalize_analysis(analysis, telemetry)
            analysis["blip_insights"] = result.get('blip_insights', {})
            analysis["blip_vqa_answers"] = result.get('blip_vqa_answers', {})
            
            final_result = {
                "frame_id": frame_id,
                "timestamp": telemetry["timestamp"],
                "location": telemetry["location"],
                **analysis,
                "model_used": "blip-gpt4o",
                "processing_time_ms": 0
            }
            
            out_path = output_dir / f"{frame_id}_analysis.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(final_result, f, indent=2)
            print(f"Analysis saved for {frame_id} (BLIP Analyzer)")
            return final_result
        except Exception as e:
            print(f"BLIP Analyzer failed, falling back to standard: {e}")
    
    # Use CLIP Analyzer if configured
    if USE_CLIP_ANALYZER:
        try:
            from src.full_enhanced_vision_analyzer import analyze_frame_full_enhanced
            print("Using CLIP + GPT-4o Analyzer...")
            result = analyze_frame_full_enhanced(image_path, telemetry)
            
            # Convert to standard format
            if result and result.get('gpt4o_enhanced', {}).get('success'):
                gpt4o_content = result['gpt4o_enhanced']['enhanced_analysis']
                try:
                    analysis = extract_json_payload(gpt4o_content)
                except Exception:
                    analysis = {"raw_response": gpt4o_content}
                
                analysis = _normalize_analysis(analysis, telemetry)
                analysis["clip_threat_score"] = result['clip_analysis']['threat_score']
                analysis["overall_threat_level"] = result['overall_threat_level']
                
                final_result = {
                    "frame_id": frame_id,
                    "timestamp": telemetry["timestamp"],
                    "location": telemetry["location"],
                    **analysis,
                    "model_used": "clip-gpt4o",
                    "processing_time_ms": 0
                }
                
                out_path = output_dir / f"{frame_id}_analysis.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(final_result, f, indent=2)
                print(f"Analysis saved for {frame_id} (CLIP Analyzer)")
                return final_result
        except Exception as e:
            print(f"CLIP Analyzer failed, falling back to standard: {e}")
    
    # Standard Gemini Vision (default)
    print("Using Standard Gemini Vision...")
    from src.unified_context import get_vlm_prompt_context
    from src.frame_preprocessor import preprocess_frame_image, quality_prompt_hints, preprocessing_enabled

    frame_history = get_vlm_prompt_context(frame_id, window=5)
    user_prompt = USER_PROMPT_TEMPLATE.replace("{telemetry}", json.dumps(telemetry, indent=2))
    user_prompt = (
        f"=== PRIOR SESSION CONTEXT ===\n{frame_history}\n\n=== CURRENT FRAME ===\n{user_prompt}"
    )
    image_input: Union[Path, bytes] = image_path
    image_quality: Dict[str, Any] = {}
    if preprocessing_enabled():
        image_bytes, image_quality = preprocess_frame_image(image_path)
        image_input = image_bytes
        user_prompt += quality_prompt_hints(image_quality)
    else:
        from src.frame_preprocessor import assess_image_quality
        image_quality = assess_image_quality(image_path)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    start = time.time()
    try:
        from src.api_retry import is_quota_exhausted_error, mark_quota_exhausted

        gemini = generate_vision(full_prompt, image_input, max_output_tokens=2048, temperature=0.2)
        if not gemini.get("success"):
            raise RuntimeError(gemini.get("error", "Gemini vision failed"))
        elapsed = int((time.time() - start) * 1000)
        content = gemini["text"]
        model_label = gemini.get("model_used", settings.GEMINI_MODEL)
        try:
            analysis = extract_json_payload(content)
        except Exception:
            print(f"Could not parse JSON for {frame_id}, saving raw content.")
            analysis = {"raw_response": content}
        analysis = _normalize_analysis(analysis, telemetry)
        analysis["alert_reasoning"] = analysis.get("alert_reasoning") or analysis.get("recommended_action", "")
        analysis["reasoning_signals"] = analysis.get("reasoning_signals", [])
        result = {
            "frame_id": frame_id,
            "timestamp": telemetry["timestamp"],
            "location": telemetry["location"],
            **analysis,
            "model_used": model_label,
            "processing_time_ms": elapsed,
            "image_quality": image_quality or None,
        }
        out_path = output_dir / f"{frame_id}_analysis.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Analysis saved for {frame_id} ({elapsed} ms)")
        return result
    except Exception as e:
        print(f"Vision analysis failed for {frame_id}: {e}")
        if is_quota_exhausted_error(e):
            mark_quota_exhausted()
        if quota_exhausted() or _use_offline_vision():
            return _save_offline_result(frame_id, image_path, telemetry, output_dir)
        return {}

def analyze_all_frames():
    """
    Runs analysis for all frames using telemetry and extracted images.
    Supports parallel VLM processing for speedup, followed by thread-safe sequential post-processing.
    """
    meta_path = settings.OUTPUTS_DIR / "extraction_log.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        frame_meta = json.load(f)["frames"]
    telemetry_path = settings.TELEMETRY_DIR / "all_telemetry.json"
    with open(telemetry_path, "r", encoding="utf-8") as f:
        all_telemetry = json.load(f)
    
    # Build telemetry lookup by frame_id to handle rejected frames
    telemetry_lookup = {t.get("frame_id", f"frame_{i+1:03d}"): t 
                         for i, t in enumerate(all_telemetry)}
    
    from src.unified_context import get_unified_context
    from concurrent.futures import ThreadPoolExecutor

    unified = get_unified_context(session_dir=settings.SESSION_DIR)
    
    # Compile a list of tasks for VLM analysis
    tasks = []
    for i, frame in enumerate(frame_meta):
        frame_id = f"frame_{i+1:03}"
        extracted_root = settings.EXTRACTED_DIR
        if not os.environ.get("SESSION_ID"):
            latest_extracted_folder = get_latest_extracted_folder()
            if latest_extracted_folder and Path(latest_extracted_folder).exists():
                extracted_root = Path(latest_extracted_folder)
        image_path = extracted_root / frame["filename"]
        
        # Get telemetry by frame_id (handles rejected frames gracefully)
        telemetry = telemetry_lookup.get(frame_id)
        if telemetry is None:
            print(f"WARNING: No telemetry for {frame_id} - frame may have been rejected")
            continue
            
        if not image_path.exists():
            print(f"ERROR: Image file not found: {image_path}")
            continue
            
        telemetry = unified.ensure_telemetry_session_id(telemetry)
        tasks.append((frame_id, image_path, telemetry))

    # Run VLM requests in parallel
    # Note: max_workers is set to 5 by default, but can be controlled via environment variable
    max_workers = int(os.environ.get("MAX_VISION_WORKERS", "5"))
    print(f"\n[VISION] Running parallel VLM analysis for {len(tasks)} frames using {max_workers} workers...")
    
    def _worker(task):
        fid, img_path, tel = task
        from src.cancellation import is_cancelled
        session_id = os.environ.get("SESSION_ID") or getattr(settings, "SESSION_ID", "")
        if is_cancelled(session_id):
            print(f"Skipping VLM analysis for {fid} - session {session_id} is cancelled")
            return fid, tel, None
        try:
            print(f"Analyzing {fid} in parallel...")
            res = analyze_frame(fid, img_path, tel)
            return fid, tel, res
        except Exception as exc:
            print(f"FAILED to analyze {fid} in parallel: {exc}")
            return fid, tel, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        completed = list(executor.map(_worker, tasks))
        
    # Sort completed tasks by frame_id to preserve chronological order
    completed.sort(key=lambda x: x[0])
    
    # Sequential, thread-safe session context updates
    all_results = []
    # Build complete dict mapping to reconstruct the original list structure with None placeholders
    result_map = {fid: (tel, res) for fid, tel, res in completed}
    
    from src.cancellation import is_cancelled
    session_id = os.environ.get("SESSION_ID") or getattr(settings, "SESSION_ID", "")
    if is_cancelled(session_id):
        raise RuntimeError("Pipeline execution cancelled by user request.")

    for i, frame in enumerate(frame_meta):
        frame_id = f"frame_{i+1:03}"
        if is_cancelled(session_id):
            raise RuntimeError("Pipeline execution cancelled by user request.")
        if frame_id not in result_map:
            # Re-insert skipped/rejected placeholders
            all_results.append(None)
            continue
            
        telemetry, result = result_map[frame_id]
        all_results.append(result)
        
        if result:
            alert_summary = derive_alert_from_analysis(result, telemetry)
            unified.record_frame(frame_id, telemetry, result, alert_summary)
            
    # Save combined
    if is_cancelled(session_id):
        raise RuntimeError("Pipeline execution cancelled by user request.")
    combined_path = settings.ANALYSIS_DIR / "all_analysis.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined analysis saved to {combined_path}")
    return all_results

if __name__ == "__main__":
    analyze_all_frames()
