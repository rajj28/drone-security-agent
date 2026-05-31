"""
behavioral_analyzer.py - Advanced behavioral threat detection for security monitoring.

Analyzes person behavior patterns to detect suspicious activities, theft indicators,
and security threats based on movement, positioning, and interaction patterns.
"""

import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
from src.config import settings

class BehavioralAnalyzer:
    """Analyzes behavioral patterns for security threat detection."""
    
    def __init__(self):
        self.theft_indicators = {
            'hand_movements': ['pocket', 'bag', 'conceal', 'hide', 'reach', 'grab', 'take'],
            'positioning': ['behind_counter', 'facing_away', 'blocking_view', 'corner_positioning'],
            'interaction_patterns': ['distracting', 'rapid_movements', 'nervous_glancing', 'avoiding_eye_contact'],
            'group_behavior': ['separating', 'creating_diversion', 'blocking_shopkeeper_view']
        }
        
        self.suspicious_behaviors = {
            'movements': ['loitering', 'pacing', 'circling', 'sudden_direction_changes'],
            'timing': ['watching_for_opportunities', 'waiting_for_moments', 'quick_actions'],
            'social': ['avoiding_staff', 'ignoring_service', 'unusual_questions']
        }
    
    def analyze_person_behavior(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual person behavior for threat indicators."""
        person_features = analysis.get('person_features', [])
        vlm_description = analysis.get('vlm_description', '').lower()
        
        behavioral_threats = []
        threat_score = 0
        
        for person in person_features:
            person_threats = self._analyze_single_person(person, vlm_description)
            behavioral_threats.extend(person_threats)
            threat_score += len(person_threats)
        
        return {
            'behavioral_threats_detected': behavioral_threats,
            'threat_score': threat_score,
            'behavioral_risk_level': self._calculate_risk_level(threat_score, len(person_features)),
            'recommendations': self._generate_recommendations(behavioral_threats)
        }
    
    def _analyze_single_person(self, person: Dict[str, Any], scene_context: str) -> List[str]:
        """Analyze a single person for behavioral threats."""
        threats = []
        
        # Check actions for theft indicators
        actions = person.get('actions', [])
        for action in actions:
            if any(indicator in action.lower() for indicator in self.theft_indicators['hand_movements']):
                threats.append(f"theft_indicator_{action}")
        
        # Check suspicious actions
        suspicious_actions = person.get('suspicious_actions', [])
        for action in suspicious_actions:
            if action in ['hand_to_pocket', 'positioning_to_hide', 'nervous_glancing']:
                threats.append(f"suspicious_behavior_{action}")
        
        # Check positioning
        position = person.get('position_in_frame', '')
        if position in ['corner', 'edge', 'behind']:
            threats.append("suspicious_positioning")
        
        # Check threat level
        threat_level = person.get('threat_level', 'none')
        if threat_level in ['medium', 'high']:
            threats.append(f"person_threat_level_{threat_level}")
        
        return threats
    
    def _calculate_risk_level(self, threat_score: int, person_count: int) -> str:
        """Calculate overall behavioral risk level."""
        if threat_score == 0:
            return 'low'
        elif threat_score <= person_count:
            return 'medium'
        else:
            return 'high'
    
    def _generate_recommendations(self, threats: List[str]) -> List[str]:
        """Generate security recommendations based on detected threats."""
        recommendations = []
        
        theft_indicators = [t for t in threats if 'theft_indicator' in t]
        if theft_indicators:
            recommendations.append("Monitor for potential theft - suspicious hand movements detected")
        
        suspicious_positioning = [t for t in threats if 'suspicious_positioning' in t]
        if suspicious_positioning:
            recommendations.append("Person positioning suggests attempt to conceal actions")
        
        nervous_behavior = [t for t in threats if 'nervous_glancing' in t]
        if nervous_behavior:
            recommendations.append("Nervous behavior detected - may indicate suspicious intent")
        
        if len(threats) >= 3:
            recommendations.append("Multiple threat indicators - immediate staff attention recommended")
        
        return recommendations
    
    def detect_shoplifting_scenario(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Specifically detect shoplifting scenarios."""
        people_count = analysis.get('people_count', 0)
        vlm_description = analysis.get('vlm_description', '').lower()
        objects_detected = analysis.get('objects_detected', [])
        
        shoplifting_indicators = []
        
        # Multiple people in retail setting
        if people_count >= 2 and any(word in vlm_description for word in ['shop', 'store', 'retail']):
            shoplifting_indicators.append("multiple_people_retail_setting")
        
        # Electronics present with multiple people
        if 'phone' in vlm_description or 'mobile' in vlm_description:
            if people_count >= 2:
                shoplifting_indicators.append("electronics_theft_risk")
        
        # Check for distraction patterns
        person_features = analysis.get('person_features', [])
        for person in person_features:
            actions = person.get('actions', [])
            if 'distracting' in actions:
                shoplifting_indicators.append("distraction_technique")
        
        return {
            'shoplifting_risk': len(shoplifting_indicators) >= 2,
            'indicators': shoplifting_indicators,
            'risk_level': 'high' if len(shoplifting_indicators) >= 3 else 'medium' if len(shoplifting_indicators) >= 1 else 'low'
        }

def analyze_behavioral_threats(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Main function to analyze behavioral threats."""
    analyzer = BehavioralAnalyzer()
    
    # General behavioral analysis
    behavioral_analysis = analyzer.analyze_person_behavior(analysis)
    
    # Specific shoplifting detection
    shoplifting_analysis = analyzer.detect_shoplifting_scenario(analysis)
    
    return {
        'behavioral_analysis': behavioral_analysis,
        'shoplifting_detection': shoplifting_analysis,
        'overall_threat_level': _calculate_overall_threat(behavioral_analysis, shoplifting_analysis)
    }

def _calculate_overall_threat(behavioral: Dict[str, Any], shoplifting: Dict[str, Any]) -> str:
    """Calculate overall threat level from all analyses."""
    behavioral_score = behavioral['threat_score']
    shoplifting_risk = shoplifting['shoplifting_risk']
    
    if shoplifting_risk or behavioral_score >= 3:
        return 'high'
    elif behavioral_score >= 1 or shoplifting['risk_level'] == 'medium':
        return 'medium'
    else:
        return 'low'
