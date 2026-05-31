"""
learning_agent.py - Intelligent learning agent for security video analysis

This agent learns from each video analysis, remembers patterns, incidents,
and improves threat detection over time through cumulative knowledge.

Features:
- Pattern recognition and learning
- Incident memory system
- Per-video learning and memory
- Cross-video pattern learning
- Adaptive threat detection
- Knowledge base management
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityPattern:
    """Security pattern learned from incidents"""
    pattern_id: str
    pattern_type: str  # "behavioral", "visual", "temporal", "contextual"
    description: str
    confidence: float
    frequency: int
    first_seen: datetime
    last_seen: datetime
    associated_threats: List[str]
    context_tags: List[str]
    success_rate: float
    false_positive_rate: float
    
    def __post_init__(self):
        if isinstance(self.first_seen, str):
            self.first_seen = datetime.fromisoformat(self.first_seen)
        if isinstance(self.last_seen, str):
            self.last_seen = datetime.fromisoformat(self.last_seen)

@dataclass
class SecurityIncident:
    """Security incident record"""
    incident_id: str
    video_id: str
    incident_type: str
    severity: str
    description: string
    timestamp: datetime
    location: str
    camera_id: str
    frames_involved: List[str]
    patterns_detected: List[str]
    outcome: str  # "confirmed_threat", "false_positive", "investigating"
    response_time: float  # Time to detect in seconds
    confidence: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

@dataclass
class VideoLearningProfile:
    """Learning profile for a specific video"""
    video_id: str
    total_frames: int
    threat_frames: int
    patterns_detected: List[str]
    unique_behaviors: Set[str]
    threat_timeline: List[Tuple[float, str]]
    confidence_trend: List[float]
    learning_score: float
    adaptation_level: int  # 0-100
    created_at: datetime
    updated_at: datetime

class LearningAgent:
    """Intelligent learning agent for security analysis"""
    
    def __init__(self, knowledge_base_path: str = "data/knowledge_base"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        
        # Knowledge storage
        self.patterns_file = self.knowledge_base_path / "patterns.json"
        self.incidents_file = self.knowledge_base_path / "incidents.json"
        self.profiles_file = self.knowledge_base_path / "video_profiles.json"
        self.knowledge_graph_file = self.knowledge_base_path / "knowledge_graph.pkl"
        
        # Learning data
        self.patterns: Dict[str, SecurityPattern] = {}
        self.incidents: Dict[str, SecurityIncident] = {}
        self.video_profiles: Dict[str, VideoLearningProfile] = {}
        self.knowledge_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Pattern recognition
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.pattern_vectors = None
        self.is_trained = False
        
        # Learning parameters
        self.min_pattern_frequency = 3
        self.confidence_threshold = 0.7
        self.adaptation_rate = 0.1
        
        # Load existing knowledge
        self.load_knowledge()
        
        logger.info(f"Learning agent initialized with {len(self.patterns)} patterns, {len(self.incidents)} incidents")
    
    def load_knowledge(self):
        """Load existing knowledge base"""
        try:
            # Load patterns
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    for pattern_id, pattern_data in patterns_data.items():
                        self.patterns[pattern_id] = SecurityPattern(**pattern_data)
            
            # Load incidents
            if self.incidents_file.exists():
                with open(self.incidents_file, 'r') as f:
                    incidents_data = json.load(f)
                    for incident_id, incident_data in incidents_data.items():
                        self.incidents[incident_id] = SecurityIncident(**incident_data)
            
            # Load video profiles
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                    for video_id, profile_data in profiles_data.items():
                        profile_data['unique_behaviors'] = set(profile_data['unique_behaviors'])
                        self.video_profiles[video_id] = VideoLearningProfile(**profile_data)
            
            # Load knowledge graph
            if self.knowledge_graph_file.exists():
                with open(self.knowledge_graph_file, 'rb') as f:
                    self.knowledge_graph = pickle.load(f)
            
            # Train pattern recognition if we have enough data
            if len(self.patterns) >= 5:
                self.train_pattern_recognition()
                
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
    
    def save_knowledge(self):
        """Save knowledge base to disk"""
        try:
            # Save patterns
            patterns_data = {}
            for pattern_id, pattern in self.patterns.items():
                pattern_dict = asdict(pattern)
                pattern_dict['first_seen'] = pattern.first_seen.isoformat()
                pattern_dict['last_seen'] = pattern.last_seen.isoformat()
                patterns_data[pattern_id] = pattern_dict
            
            with open(self.patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save incidents
            incidents_data = {}
            for incident_id, incident in self.incidents.items():
                incident_dict = asdict(incident)
                incident_dict['timestamp'] = incident.timestamp.isoformat()
                incidents_data[incident_id] = incident_dict
            
            with open(self.incidents_file, 'w') as f:
                json.dump(incidents_data, f, indent=2)
            
            # Save video profiles
            profiles_data = {}
            for video_id, profile in self.video_profiles.items():
                profile_dict = asdict(profile)
                profile_dict['unique_behaviors'] = list(profile.unique_behaviors)
                profile_dict['created_at'] = profile.created_at.isoformat()
                profile_dict['updated_at'] = profile.updated_at.isoformat()
                profiles_data[video_id] = profile_dict
            
            with open(self.profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            # Save knowledge graph
            with open(self.knowledge_graph_file, 'wb') as f:
                pickle.dump(dict(self.knowledge_graph), f)
            
            logger.info("Knowledge base saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
    
    def analyze_video_for_learning(self, video_id: str, frame_analyses: List[Dict[str, Any]], 
                                  video_metadata: Dict[str, Any]) -> VideoLearningProfile:
        """Analyze video to extract learning patterns"""
        logger.info(f"Analyzing video {video_id} for learning patterns")
        
        # Extract basic statistics
        total_frames = len(frame_analyses)
        threat_frames = len([f for f in frame_analyses if f.get('threat_level', '').lower() in ['high', 'medium']])
        
        # Extract patterns from frame analyses
        patterns_detected = []
        unique_behaviors = set()
        threat_timeline = []
        confidence_trend = []
        
        for frame in frame_analyses:
            # Extract behaviors from analysis
            analysis_result = frame.get('analysis_result', {})
            
            # Extract behavioral patterns
            if 'gpt4o_enhanced' in analysis_result:
                gpt4o_result = analysis_result['gpt4o_enhanced']
                if 'enhanced_analysis' in gpt4o_result:
                    enhanced_text = str(gpt4o_result['enhanced_analysis'])
                    
                    # Extract behaviors using regex patterns
                    behaviors = self.extract_behaviors(enhanced_text)
                    unique_behaviors.update(behaviors)
                    
                    # Extract patterns
                    frame_patterns = self.extract_frame_patterns(enhanced_text)
                    patterns_detected.extend(frame_patterns)
            
            # Build threat timeline
            if frame.get('threat_level', '').lower() in ['high', 'medium']:
                threat_timeline.append((frame.get('timestamp', 0), frame.get('threat_level', '')))
            
            # Track confidence
            confidence_trend.append(frame.get('confidence', 0.0))
        
        # Create or update video learning profile
        if video_id in self.video_profiles:
            profile = self.video_profiles[video_id]
            profile.total_frames = total_frames
            profile.threat_frames = threat_frames
            profile.patterns_detected = list(set(patterns_detected + profile.patterns_detected))
            profile.unique_behaviors.update(unique_behaviors)
            profile.threat_timeline = threat_timeline
            profile.confidence_trend = confidence_trend
            profile.updated_at = datetime.utcnow()
        else:
            profile = VideoLearningProfile(
                video_id=video_id,
                total_frames=total_frames,
                threat_frames=threat_frames,
                patterns_detected=list(set(patterns_detected)),
                unique_behaviors=unique_behaviors,
                threat_timeline=threat_timeline,
                confidence_trend=confidence_trend,
                learning_score=0.0,
                adaptation_level=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        
        # Calculate learning score and adaptation level
        profile.learning_score = self.calculate_learning_score(profile)
        profile.adaptation_level = self.calculate_adaptation_level(profile)
        
        # Save profile
        self.video_profiles[video_id] = profile
        
        # Learn patterns from this video
        self.learn_from_video(video_id, frame_analyses, video_metadata)
        
        return profile
    
    def extract_behaviors(self, text: str) -> List[str]:
        """Extract behavioral patterns from text"""
        behaviors = []
        
        # Common behavioral patterns
        behavior_patterns = [
            r'(person|individual|subject)\s+(is|appears|seems)\s+(reaching|grabbing|taking|stealing|concealing|hiding)',
            r'(looking|scanning|surveying)\s+(around|nervously|suspiciously)',
            r'(hand|hands)\s+(in|into|near)\s+(pocket|bag|purse|backpack)',
            r'(quick|sudden|rapid)\s+(movement|motion|action)',
            r'(avoiding|evading)\s+(eye\s+contact|attention|scrutiny)',
            r'(multiple|several|group)\s+(people|individuals)\s+(gathering|crowding|loitering)',
            r'(checking|watching|monitoring)\s+(for|if|whether)',
            r'(positioning|moving)\s+(to|towards|away\s+from)',
        ]
        
        for pattern in behavior_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    behaviors.append(' '.join(match))
                else:
                    behaviors.append(match)
        
        return list(set(behaviors))
    
    def extract_frame_patterns(self, text: str) -> List[str]:
        """Extract security patterns from frame analysis"""
        patterns = []
        
        # Pattern keywords
        pattern_keywords = {
            'theft_pattern': ['theft', 'stealing', 'shoplifting', 'taking', 'grabbing'],
            'concealment_pattern': ['concealing', 'hiding', 'pocket', 'bag', 'concealed'],
            'surveillance_pattern': ['watching', 'scanning', 'surveying', 'monitoring'],
            'nervous_behavior': ['nervous', 'anxious', 'tense', 'fidgeting'],
            'suspicious_movement': ['suspicious', 'unusual', 'abnormal', 'strange'],
            'group_behavior': ['multiple', 'group', 'together', 'coordinated'],
            'distraction_pattern': ['distracting', 'diverting', 'misdirection'],
            'opportunity_pattern': ['opportunity', 'unattended', 'unprotected', 'vulnerable']
        }
        
        text_lower = text.lower()
        
        for pattern_name, keywords in pattern_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                patterns.append(pattern_name)
        
        return patterns
    
    def learn_from_video(self, video_id: str, frame_analyses: List[Dict[str, Any]], 
                          video_metadata: Dict[str, Any]):
        """Learn patterns from video analysis"""
        logger.info(f"Learning from video {video_id}")
        
        # Analyze threat patterns
        threat_frames = [f for f in frame_analyses if f.get('threat_level', '').lower() in ['high', 'medium']]
        
        for frame in threat_frames:
            # Create incident record
            incident_id = f"incident_{video_id}_{frame.get('frame_number', 0)}"
            
            # Extract patterns from frame
            analysis_result = frame.get('analysis_result', {})
            patterns_detected = []
            
            if 'gpt4o_enhanced' in analysis_result:
                gpt4o_result = analysis_result['gpt4o_enhanced']
                if 'enhanced_analysis' in gpt4o_result:
                    enhanced_text = str(gpt4o_result['enhanced_analysis'])
                    patterns_detected = self.extract_frame_patterns(enhanced_text)
            
            # Create incident
            incident = SecurityIncident(
                incident_id=incident_id,
                video_id=video_id,
                incident_type="threat_detected",
                severity=frame.get('threat_level', 'medium'),
                description=f"Threat detected at frame {frame.get('frame_number', 0)}",
                timestamp=datetime.utcnow(),
                location=video_metadata.get('location', 'unknown'),
                camera_id=video_metadata.get('camera_id', 'unknown'),
                frames_involved=[frame.get('frame_id', '')],
                patterns_detected=patterns_detected,
                outcome="investigating",
                response_time=frame.get('timestamp', 0.0),
                confidence=frame.get('confidence', 0.0),
                metadata={
                    'frame_number': frame.get('frame_number', 0),
                    'people_detected': frame.get('people_detected', 0),
                    'objects_detected': frame.get('objects_detected', [])
                }
            )
            
            self.incidents[incident_id] = incident
            
            # Update pattern frequency
            for pattern_name in patterns_detected:
                self.update_pattern_frequency(pattern_name, incident)
        
        # Update knowledge graph
        self.update_knowledge_graph(video_id, frame_analyses)
        
        # Retrain pattern recognition
        if len(self.patterns) >= 5:
            self.train_pattern_recognition()
    
    def update_pattern_frequency(self, pattern_name: str, incident: SecurityIncident):
        """Update pattern frequency and statistics"""
        if pattern_name not in self.patterns:
            # Create new pattern
            pattern_id = f"pattern_{pattern_name}_{int(datetime.utcnow().timestamp())}"
            
            self.patterns[pattern_id] = SecurityPattern(
                pattern_id=pattern_id,
                pattern_type="behavioral",
                description=f"Pattern: {pattern_name}",
                confidence=incident.confidence,
                frequency=1,
                first_seen=incident.timestamp,
                last_seen=incident.timestamp,
                associated_threats=[incident.severity],
                context_tags=[incident.location, incident.camera_id],
                success_rate=1.0,
                false_positive_rate=0.0
            )
        else:
            # Update existing pattern
            pattern = self.patterns[pattern_name]
            pattern.frequency += 1
            pattern.last_seen = incident.timestamp
            pattern.confidence = (pattern.confidence + incident.confidence) / 2
            
            if incident.severity not in pattern.associated_threats:
                pattern.associated_threats.append(incident.severity)
            
            if incident.location not in pattern.context_tags:
                pattern.context_tags.append(incident.location)
    
    def update_knowledge_graph(self, video_id: str, frame_analyses: List[Dict[str, Any]]):
        """Update knowledge graph with relationships"""
        # Connect videos with similar patterns
        video_patterns = set()
        for frame in frame_analyses:
            analysis_result = frame.get('analysis_result', {})
            if 'gpt4o_enhanced' in analysis_result:
                gpt4o_result = analysis_result['gpt4o_enhanced']
                if 'enhanced_analysis' in gpt4o_result:
                    patterns = self.extract_frame_patterns(str(gpt4o_result['enhanced_analysis']))
                    video_patterns.update(patterns)
        
        # Add connections to knowledge graph
        for pattern in video_patterns:
            self.knowledge_graph[pattern].append(video_id)
        
        # Find similar videos
        for other_video_id, other_profile in self.video_profiles.items():
            if other_video_id != video_id:
                similarity = self.calculate_video_similarity(video_id, other_video_id)
                if similarity > 0.7:  # High similarity threshold
                    self.knowledge_graph[f"similar_videos_{video_id}"].append(other_video_id)
    
    def calculate_video_similarity(self, video_id1: str, video_id2: str) -> float:
        """Calculate similarity between two videos"""
        if video_id1 not in self.video_profiles or video_id2 not in self.video_profiles:
            return 0.0
        
        profile1 = self.video_profiles[video_id1]
        profile2 = self.video_profiles[video_id2]
        
        # Calculate pattern overlap
        patterns1 = set(profile1.patterns_detected)
        patterns2 = set(profile2.patterns_detected)
        
        if not patterns1 or not patterns2:
            return 0.0
        
        intersection = len(patterns1.intersection(patterns2))
        union = len(patterns1.union(patterns2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_learning_score(self, profile: VideoLearningProfile) -> float:
        """Calculate learning score for a video profile"""
        if profile.total_frames == 0:
            return 0.0
        
        # Factors for learning score
        threat_ratio = profile.threat_frames / profile.total_frames
        pattern_diversity = len(profile.patterns_detected)
        behavior_diversity = len(profile.unique_behaviors)
        
        # Confidence trend analysis
        if profile.confidence_trend:
            avg_confidence = sum(profile.confidence_trend) / len(profile.confidence_trend)
            confidence_stability = 1.0 - (max(profile.confidence_trend) - min(profile.confidence_trend))
        else:
            avg_confidence = 0.0
            confidence_stability = 0.0
        
        # Calculate weighted score
        learning_score = (
            threat_ratio * 0.3 +
            min(pattern_diversity / 10, 1.0) * 0.3 +
            min(behavior_diversity / 20, 1.0) * 0.2 +
            avg_confidence * 0.1 +
            confidence_stability * 0.1
        )
        
        return min(learning_score, 1.0)
    
    def calculate_adaptation_level(self, profile: VideoLearningProfile) -> int:
        """Calculate adaptation level (0-100)"""
        base_level = int(profile.learning_score * 100)
        
        # Bonus factors
        pattern_bonus = min(len(profile.patterns_detected) * 2, 20)
        behavior_bonus = min(len(profile.unique_behaviors), 20)
        
        adaptation_level = min(base_level + pattern_bonus + behavior_bonus, 100)
        
        return adaptation_level
    
    def train_pattern_recognition(self):
        """Train pattern recognition model"""
        try:
            if len(self.patterns) < 5:
                logger.warning("Not enough patterns to train recognition model")
                return
            
            # Prepare training data
            pattern_texts = []
            pattern_labels = []
            
            for pattern_id, pattern in self.patterns.items():
                pattern_texts.append(pattern.description)
                pattern_labels.append(pattern.pattern_type)
            
            # Train vectorizer
            self.pattern_vectors = self.vectorizer.fit_transform(pattern_texts)
            self.is_trained = True
            
            logger.info(f"Pattern recognition trained with {len(pattern_texts)} patterns")
            
        except Exception as e:
            logger.error(f"Failed to train pattern recognition: {e}")
    
    def recognize_patterns(self, text: str) -> List[Tuple[str, float]]:
        """Recognize patterns in new text"""
        if not self.is_trained:
            return []
        
        try:
            # Vectorize input text
            text_vector = self.vectorizer.transform([text])
            
            # Calculate similarities
            similarities = cosine_similarity(text_vector, self.pattern_vectors)[0]
            
            # Get top matches
            pattern_ids = list(self.patterns.keys())
            results = []
            
            for i, similarity in enumerate(similarities):
                if similarity > 0.3:  # Minimum similarity threshold
                    pattern_id = pattern_ids[i]
                    pattern = self.patterns[pattern_id]
                    results.append((pattern.pattern_type, similarity))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:5]  # Return top 5 matches
            
        except Exception as e:
            logger.error(f"Pattern recognition failed: {e}")
            return []
    
    def get_adaptive_analysis(self, video_id: str, frame_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get adaptive analysis based on learned patterns"""
        if video_id not in self.video_profiles:
            return frame_analysis
        
        profile = self.video_profiles[video_id]
        
        # Extract text from analysis
        analysis_text = ""
        if 'gpt4o_enhanced' in frame_analysis.get('analysis_result', {}):
            gpt4o_result = frame_analysis['analysis_result']['gpt4o_enhanced']
            if 'enhanced_analysis' in gpt4o_result:
                analysis_text = str(gpt4o_result['enhanced_analysis'])
        
        # Recognize patterns
        recognized_patterns = self.recognize_patterns(analysis_text)
        
        # Get similar video insights
        similar_videos = self.knowledge_graph.get(f"similar_videos_{video_id}", [])
        
        # Adaptive confidence adjustment
        base_confidence = frame_analysis.get('confidence', 0.0)
        adaptive_confidence = base_confidence
        
        # Boost confidence if patterns match learned threats
        for pattern_type, similarity in recognized_patterns:
            if pattern_type in ['theft_pattern', 'concealment_pattern', 'suspicious_movement']:
                adaptive_confidence += similarity * 0.1
        
        # Adjust based on video learning score
        adaptive_confidence *= (1.0 + profile.learning_score * 0.2)
        
        # Cap confidence
        adaptive_confidence = min(adaptive_confidence, 1.0)
        
        return {
            'original_analysis': frame_analysis,
            'recognized_patterns': recognized_patterns,
            'similar_videos': similar_videos,
            'adaptive_confidence': adaptive_confidence,
            'learning_score': profile.learning_score,
            'adaptation_level': profile.adaptation_level,
            'video_profile': {
                'total_frames': profile.total_frames,
                'threat_frames': profile.threat_frames,
                'patterns_detected': profile.patterns_detected,
                'unique_behaviors': list(profile.unique_behaviors)
            }
        }
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights"""
        insights = {
            'total_patterns': len(self.patterns),
            'total_incidents': len(self.incidents),
            'total_videos': len(self.video_profiles),
            'pattern_types': {},
            'top_patterns': [],
            'learning_trends': {},
            'knowledge_graph_size': len(self.knowledge_graph),
            'adaptation_summary': {}
        }
        
        # Pattern type distribution
        pattern_types = Counter()
        for pattern in self.patterns.values():
            pattern_types[pattern.pattern_type] += 1
        
        insights['pattern_types'] = dict(pattern_types)
        
        # Top patterns by frequency
        top_patterns = sorted(self.patterns.values(), key=lambda x: x.frequency, reverse=True)[:10]
        insights['top_patterns'] = [
            {
                'pattern_id': p.pattern_id,
                'description': p.description,
                'frequency': p.frequency,
                'confidence': p.confidence
            }
            for p in top_patterns
        ]
        
        # Learning trends
        if self.video_profiles:
            avg_learning_score = sum(p.learning_score for p in self.video_profiles.values()) / len(self.video_profiles)
            avg_adaptation_level = sum(p.adaptation_level for p in self.video_profiles.values()) / len(self.video_profiles)
            
            insights['learning_trends'] = {
                'avg_learning_score': avg_learning_score,
                'avg_adaptation_level': avg_adaptation_level,
                'total_behaviors_learned': sum(len(p.unique_behaviors) for p in self.video_profiles.values())
            }
        
        # Adaptation summary
        adaptation_levels = [p.adaptation_level for p in self.video_profiles.values()]
        if adaptation_levels:
            insights['adaptation_summary'] = {
                'high_adaptation_videos': len([x for x in adaptation_levels if x >= 80]),
                'medium_adaptation_videos': len([x for x in adaptation_levels if 50 <= x < 80]),
                'low_adaptation_videos': len([x for x in adaptation_levels if x < 50])
            }
        
        return insights
    
    def update_incident_outcome(self, incident_id: str, outcome: str):
        """Update incident outcome for learning"""
        if incident_id in self.incidents:
            incident = self.incidents[incident_id]
            incident.outcome = outcome
            
            # Update pattern statistics
            for pattern_name in incident.pattern_detected:
                # Find pattern by name (this is simplified - in practice, you'd have better mapping)
                for pattern in self.patterns.values():
                    if pattern_name in pattern.description.lower():
                        if outcome == "false_positive":
                            pattern.false_positive_rate += 0.1
                        elif outcome == "confirmed_threat":
                            pattern.success_rate += 0.1
                        break
            
            # Save updated knowledge
            self.save_knowledge()
            
            logger.info(f"Updated incident {incident_id} outcome to {outcome}")

# Global learning agent instance
learning_agent = LearningAgent()
