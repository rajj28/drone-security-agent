"""
object_continuity.py — Cross-frame object-continuity heuristic.

Each frame is scored independently by the VLM, so an object that is being
actively handled in frame N and is simply gone from frame N+1 — tucked into
a bag, pocket, or under clothing — leaves no visual trace for a single-frame
analysis to catch. Frame N+1 just looks like a calm person doing nothing
suspicious, because by then the concealment already happened off-camera
(between the two capture timestamps).

This module surfaces that "handled object vanished" pattern as an explicit,
deterministic signal, instead of relying on the VLM to infer it from prose it
was never told to compare across frames.

IMPORTANT precision note: an earlier version of this diffed the entire
objects_detected list between consecutive frames. That fired on nearly every
frame pair in testing — a VLM's per-frame object list is not stable enough for
a raw diff; attention drifts (a spoon, a window, "trees" mentioned once and
never again) with nothing suspicious happening. To keep this usable instead of
crying wolf constantly, it only tracks items explicitly described as being
HELD/MANIPULATED by a person (parsed from person_features/activity text), and
excludes common food/utensil words that change every frame during normal
eating. This trades some recall (it can still miss items the VLM never
described as "held") for the precision needed to be worth alerting on at all.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Scene furniture / worn clothing / structural elements — persist across
# frames or belong to the person, so their disappearance is not meaningful.
_BACKGROUND_TERMS = (
    "person", "people", "chair", "stool", "table", "desk", "window", "curtain",
    "wall", "floor", "rug", "carpet", "notebook", "paper", "cable", "monitor",
    "screen", "glasses", "shirt", "t-shirt", "hoodie", "jacket", "apron",
    "necklace", "shoe", "pant", "vehicle", "car", "sedan", "truck", "door",
    "gate", "fence", "sign", "light", "lamp", "counter", "shelf", "rack",
    "bin", "camera", "ceiling", "tile", "book", "tree", "trees", "sky", "grass",
)

# Words that show up and disappear constantly during ordinary eating/drinking —
# excluded outright so they never register as a "held item" candidate.
_EXCLUDED_HELD_TERMS = (
    "spoon", "fork", "knife", "food", "bowl", "plate", "cup", "utensil", "meal",
    "bite", "mouth", "napkin", "glass of water", "drink",
)

_SET_DOWN_PHRASES = ("set down", "put back", "placed on", "placed back", "returned", "left on", "put down", "handed")
_RECEPTACLE_TERMS = ("bag", "pocket", "waistband", "under", "clothing", "jacket", "hoodie", "shirt", "backpack", "purse")
_IDLE_HANDS_TERMS = ("resting", "empty", "idle", "no longer holding", "hands are down", "hands at")

# The VLM often re-describes the same physical object with a different noun
# across frames (e.g. "bottle" in one frame, "tube" the next). Group common
# synonyms so a rename isn't mistaken for the object vanishing.
_SYNONYM_GROUPS = (
    {"bottle", "tube", "container", "jar", "flask", "canister", "vial"},
    {"phone", "smartphone", "mobile", "cellphone", "iphone"},
    {"bag", "backpack", "purse", "handbag", "pouch", "satchel"},
    {"box", "package", "carton", "parcel", "crate"},
    {"wallet", "billfold"},
    {"jewelry", "necklace", "bracelet", "ring", "watch"},
)

# Objects too large to plausibly conceal under clothing/in a bag — a wording
# hedge only, still worth flagging (it may have been taken/moved) but not
# worth claiming "concealment" outright.
_BULKY_TERMS = ("bowl", "monitor", "laptop", "bag", "backpack", "box", "crate", "stool", "chair")

_SEVERITY_ORDER = {"CLEAR": 0, "UNKNOWN": 0, "NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

_HELD_ITEM_PATTERN = re.compile(
    r"(?:(?:holding|holds|held|manipulat\w*|examin\w*|grasp\w*)(?:\s+and\s+|\s*,\s*)?)+"
    r"(?:a|an|the|both|their)?\s*"
    r"([a-z][a-z0-9 '\-]{1,50}?)"
    r"(?=[.,;]| which | that | while | and looking| and check| and glanc|$)",
    re.IGNORECASE,
)


def _normalize(term: str) -> str:
    return re.sub(r"\(.*?\)", "", term).strip().lower()


def _canonical(term: str) -> str:
    """Collapses common object synonyms to one canonical token so a rename
    across frames (e.g. "bottle" -> "tube") isn't read as a disappearance."""
    lowered = _normalize(term)
    words = set(re.findall(r"[a-z]+", lowered))
    for group in _SYNONYM_GROUPS:
        if words & group:
            return sorted(group)[0]
    return lowered


def _is_background(term: str) -> bool:
    lowered = _normalize(term)
    return any(bg in lowered for bg in _BACKGROUND_TERMS)


def _is_excluded_held_term(term: str) -> bool:
    lowered = _normalize(term)
    return any(ex in lowered for ex in _EXCLUDED_HELD_TERMS)


def _is_bulky(term: str) -> bool:
    lowered = _normalize(term)
    return any(b in lowered for b in _BULKY_TERMS)


def _text_blob(frame: Dict[str, Any]) -> str:
    return " ".join([
        str(frame.get("activity", "")),
        str(frame.get("vlm_description", "")),
        " ".join(str(p) for p in (frame.get("person_features") or [])),
    ]).lower()


def _held_candidates(frame: Dict[str, Any]) -> List[str]:
    """Extracts noun phrases explicitly described as held/manipulated by a
    person in this frame — the anchor for tracking a specific item across
    frames, as opposed to anything merely visible in the scene."""
    text = " ".join([
        str(frame.get("activity", "")),
        " ".join(str(p) for p in (frame.get("person_features") or [])),
    ])
    candidates = []
    for match in _HELD_ITEM_PATTERN.finditer(text):
        phrase = match.group(1).strip()
        if not phrase or _is_background(phrase) or _is_excluded_held_term(phrase):
            continue
        candidates.append(phrase)
    return candidates


def find_vanished_handled_objects(prev: Dict[str, Any], curr: Dict[str, Any]) -> List[str]:
    """Returns held-item phrases from `prev` that are no longer present in `curr`.

    Only items explicitly described as being held/manipulated in `prev` are
    considered (see `_held_candidates`) — this is deliberately narrower than
    diffing the full objects_detected list, which is too noisy across frames
    to use as a reliable signal (see module docstring).
    """
    # Guard against comparing across a scene change (frame with nobody in it).
    prev_people = int(prev.get("people_count", 0) or 0)
    curr_people = int(curr.get("people_count", 0) or 0)
    if prev_people == 0 or curr_people == 0:
        return []

    candidates = _held_candidates(prev)
    if not candidates:
        return []

    curr_text = _text_blob(curr)
    curr_objects = curr.get("objects_detected") or []
    curr_canonical = {_canonical(o) for o in curr_objects}

    vanished = []
    for phrase in candidates:
        canon = _canonical(phrase)
        if canon in curr_canonical:
            continue  # still present, possibly just re-described with a different noun
        if canon in curr_text:
            continue
        # Loose fallback: any significant word from the phrase still mentioned
        # anywhere in the current frame's text is treated as "still around".
        words = [w for w in re.findall(r"[a-z]+", canon) if len(w) > 3]
        if words and any(w in curr_text for w in words):
            continue
        if any(p in curr_text for p in _SET_DOWN_PHRASES):
            continue
        vanished.append(phrase)
    return vanished


def concealment_signal(prev: Dict[str, Any], curr: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Returns a concealment-signal dict if `curr` looks like it followed a
    concealment action relative to `prev`, else None.

    Requires positive evidence in `curr` — a receptacle mention (bag/pocket/
    clothing) or an explicit "hands are now empty/idle" description — not just
    the absence of a re-mention, since the VLM doesn't re-describe every
    object every frame even when nothing happened.
    """
    vanished = find_vanished_handled_objects(prev, curr)
    if not vanished:
        return None

    curr_text = _text_blob(curr)
    near_receptacle = any(term in curr_text for term in _RECEPTACLE_TERMS)
    idle_hands = any(term in curr_text for term in _IDLE_HANDS_TERMS)
    if not (near_receptacle or idle_hands):
        return None

    concealable = [v for v in vanished if not _is_bulky(v)]
    phrasing = (
        "consistent with concealment under clothing/in a bag"
        if concealable
        else "it may have been taken, concealed, or moved out of frame"
    )
    return {
        "vanished_objects": vanished,
        "escalate_to": "HIGH" if (near_receptacle and concealable) else "MEDIUM",
        "reason": (
            f"Object continuity: {', '.join(vanished)} were being actively handled in the "
            f"previous frame and are no longer visible here, with no sign they were set down "
            f"or handed off in view — {phrasing}."
        ),
    }


def escalate_severity(current: str, target: str) -> str:
    """Returns whichever of `current`/`target` is more severe."""
    current = (current or "CLEAR").upper()
    target = (target or "CLEAR").upper()
    if _SEVERITY_ORDER.get(target, 0) > _SEVERITY_ORDER.get(current, 0):
        return target
    return current
