"""
person_tracker.py — Advanced person tracking across security frames.

- Tracks individuals across frames using visual attributes
- Maintains person database with detailed descriptions
- Provides person re-identification and movement patterns
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from src.config import settings

@dataclass
class PersonSighting:
    """Represents a single sighting of a person in a frame."""
    frame_id: str
    timestamp: str
    location: str
    position_in_frame: str
    confidence: float
    actions: List[str]

@dataclass
class PersonProfile:
    """Represents a tracked person with their attributes and sightings."""
    person_id: str
    clothing_color: str
    clothing_type: str
    body_type: str
    height_estimate: str
    distinctive_features: List[str]
    first_seen: str
    last_seen: str
    total_sightings: int
    locations_visited: List[str]
    sighting_history: List[PersonSighting]
    threat_level: str = "low"
    notes: str = ""

class PersonTracker:
    """Advanced person tracking system for security monitoring."""
    
    def __init__(self):
        self.persons_db_path = settings.SESSION_DIR / "persons_database.json"
        self.persons_db: Dict[str, PersonProfile] = {}
        self.load_database()
    
    def load_database(self):
        """Load existing person database."""
        if self.persons_db_path.exists():
            try:
                with open(self.persons_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for person_id, person_data in data.items():
                        # Convert sighting history
                        sightings = [
                            PersonSighting(**s) for s in person_data.get('sighting_history', [])
                        ]
                        person_data['sighting_history'] = sightings
                        self.persons_db[person_id] = PersonProfile(**person_data)
            except Exception as e:
                print(f"Error loading person database: {e}")
    
    def save_database(self):
        """Save person database to disk."""
        try:
            data = {}
            for person_id, person in self.persons_db.items():
                person_dict = asdict(person)
                # Convert sightings back to dict
                person_dict['sighting_history'] = [
                    asdict(sighting) for sighting in person.sighting_history
                ]
                data[person_id] = person_dict
            
            with open(self.persons_db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving person database: {e}")
    
    def _calculate_similarity(self, person1: Dict[str, Any], person2: Dict[str, Any]) -> float:
        """Calculate similarity between two person descriptions."""
        similarity = 0.0
        factors = 0
        
        # Clothing color match
        if person1.get('clothing_color') and person2.get('clothing_color'):
            if person1['clothing_color'].lower() == person2['clothing_color'].lower():
                similarity += 0.4
            factors += 1
        
        # Body type match
        if person1.get('body_type') and person2.get('body_type'):
            if person1['body_type'] == person2['body_type']:
                similarity += 0.2
            factors += 1
        
        # Height match
        if person1.get('height_estimate') and person2.get('height_estimate'):
            if person1['height_estimate'] == person2['height_estimate']:
                similarity += 0.15
            factors += 1
        
        # Distinctive features overlap
        features1 = set(person1.get('distinctive_features', []))
        features2 = set(person2.get('distinctive_features', []))
        if features1 and features2:
            overlap = len(features1.intersection(features2))
            union = len(features1.union(features2))
            if union > 0:
                similarity += 0.25 * (overlap / union)
            factors += 1
        
        return similarity / factors if factors > 0 else 0.0
    
    def _find_matching_person(self, person_desc: Dict[str, Any]) -> Optional[str]:
        """Find existing person that matches the description."""
        best_match = None
        best_similarity = 0.0
        
        for person_id, person_profile in self.persons_db.items():
            # Compare with person profile
            profile_dict = asdict(person_profile)
            similarity = self._calculate_similarity(person_desc, profile_dict)
            
            if similarity > best_similarity and similarity > 0.6:  # 60% threshold
                best_similarity = similarity
                best_match = person_id
        
        return best_match
    
    def add_person_sighting(self, frame_id: str, analysis: Dict[str, Any], telemetry: Dict[str, Any]) -> List[str]:
        """Add person sightings from a frame analysis."""
        person_ids = []
        
        for person_feature in analysis.get('person_features', []):
            try:
                # Extract person attributes
                person_desc = {
                    'clothing_color': person_feature.get('clothing_color', 'unknown'),
                    'clothing_type': person_feature.get('clothing_type', 'unknown'),
                    'body_type': person_feature.get('body_type', 'average'),
                    'height_estimate': person_feature.get('height_estimate', 'average'),
                    'distinctive_features': person_feature.get('distinctive_features', [])
                }
                
                # Try to match existing person
                person_id = self._find_matching_person(person_desc)
                
                if not person_id:
                    # Create new person profile
                    person_id = f"P{len(self.persons_db) + 1:03d}"
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    new_person = PersonProfile(
                        person_id=person_id,
                        clothing_color=person_desc['clothing_color'],
                        clothing_type=person_desc['clothing_type'],
                        body_type=person_desc['body_type'],
                        height_estimate=person_desc['height_estimate'],
                        distinctive_features=person_desc['distinctive_features'],
                        first_seen=current_time,
                        last_seen=current_time,
                        total_sightings=0,
                        locations_visited=[],
                        sighting_history=[]
                    )
                    self.persons_db[person_id] = new_person
                
                # Update person profile
                person_profile = self.persons_db[person_id]
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Add sighting
                sighting = PersonSighting(
                    frame_id=frame_id,
                    timestamp=telemetry.get('timestamp', current_time),
                    location=telemetry.get('location', 'unknown'),
                    position_in_frame=person_feature.get('position_in_frame', 'center'),
                    confidence=person_feature.get('confidence', 0.5),
                    actions=person_feature.get('actions', [])
                )
                
                person_profile.sighting_history.append(sighting)
                person_profile.last_seen = current_time
                person_profile.total_sightings += 1
                
                # Update locations visited
                location = telemetry.get('location', 'unknown')
                if location not in person_profile.locations_visited:
                    person_profile.locations_visited.append(location)
                
                person_ids.append(person_id)
                
            except Exception as e:
                print(f"Error processing person in {frame_id}: {e}")
        
        return person_ids
    
    def get_person_summary(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed summary of a tracked person."""
        if person_id not in self.persons_db:
            return None
        
        person = self.persons_db[person_id]
        return {
            'person_id': person.person_id,
            'clothing_color': person.clothing_color,
            'clothing_type': person.clothing_type,
            'body_type': person.body_type,
            'height_estimate': person.height_estimate,
            'distinctive_features': person.distinctive_features,
            'first_seen': person.first_seen,
            'last_seen': person.last_seen,
            'total_sightings': person.total_sightings,
            'locations_visited': person.locations_visited,
            'threat_level': person.threat_level,
            'notes': person.notes,
            'recent_activity': person.sighting_history[-5:] if person.sighting_history else []
        }
    
    def get_all_persons_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked persons."""
        return {
            'total_persons_tracked': len(self.persons_db),
            'active_persons_today': len([p for p in self.persons_db.values() 
                                      if p.last_seen.startswith(datetime.now().strftime("%Y-%m-%d"))]),
            'persons': [self.get_person_summary(pid) for pid in self.persons_db.keys()]
        }
    
    def find_person_by_description(self, description: str) -> List[Dict[str, Any]]:
        """Find persons matching a description."""
        matching_persons = []
        desc_lower = description.lower()
        
        for person_id, person in self.persons_db.items():
            match_score = 0
            
            # Check clothing color
            if person.clothing_color.lower() in desc_lower:
                match_score += 3
            
            # Check body type
            if person.body_type.lower() in desc_lower:
                match_score += 2
            
            # Check distinctive features
            for feature in person.distinctive_features:
                if feature.lower() in desc_lower:
                    match_score += 2
            
            # Check locations
            for location in person.locations_visited:
                if location.lower() in desc_lower:
                    match_score += 1
            
            if match_score >= 2:  # Minimum threshold
                summary = self.get_person_summary(person_id)
                summary['match_score'] = match_score
                matching_persons.append(summary)
        
        # Sort by match score
        matching_persons.sort(key=lambda x: x['match_score'], reverse=True)
        return matching_persons

def process_all_frames():
    """Process all frames to build person tracking database."""
    tracker = PersonTracker()
    print("Building person tracking database...")
    
    # Get all analysis files
    analysis_files = sorted(settings.ANALYSIS_DIR.glob("frame_*_analysis.json"))
    telemetry_files = sorted(settings.TELEMETRY_DIR.glob("frame_*_telemetry.json"))
    
    print(f"Found {len(analysis_files)} analysis files")
    
    processed_count = 0
    for analysis_file in analysis_files:
        frame_id = analysis_file.stem.replace("_analysis", "")
        
        try:
            # Load analysis and telemetry
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            telemetry_file = settings.TELEMETRY_DIR / f"{frame_id}_telemetry.json"
            with open(telemetry_file, 'r', encoding='utf-8') as f:
                telemetry = json.load(f)
            
            # Add person sightings
            person_ids = tracker.add_person_sighting(frame_id, analysis, telemetry)
            if person_ids:
                print(f"Processed {frame_id}: Found persons {person_ids}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {frame_id}: {e}")
    
    # Save database
    tracker.save_database()
    
    # Print summary
    summary = tracker.get_all_persons_summary()
    print(f"\nPerson Tracking Summary:")
    print(f"   Total persons tracked: {summary['total_persons_tracked']}")
    print(f"   Active persons today: {summary['active_persons_today']}")
    print(f"   Frames processed: {processed_count}")
    
    return tracker

if __name__ == "__main__":
    tracker = process_all_frames()
