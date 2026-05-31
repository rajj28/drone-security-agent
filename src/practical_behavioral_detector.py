"""
practical_behavioral_detector.py - Rule-based behavioral threat detection that works
without relying on conservative LLM responses.

Uses scene context, object detection, and basic patterns to identify potential threats.
"""

import json
from typing import Dict, Any, List
from pathlib import Path

class PracticalBehavioralDetector:
    """Practical behavioral threat detection based on rules and patterns."""
    
    def __init__(self):
        # High-risk scenarios for theft
        self.high_risk_scenarios = {
            'retail_theft': {
                'keywords': ['shop', 'store', 'retail', 'market', 'counter'],
                'objects': ['phone', 'mobile', 'electronics', 'jewelry', 'cash'],
                'min_people': 2,
                'risk_multiplier': 2.0
            },
            'electronics_theft': {
                'keywords': ['phone', 'mobile', 'electronics', 'gadget'],
                'objects': ['phone', 'mobile', 'tablet', 'laptop'],
                'min_people': 2,
                'risk_multiplier': 1.8
            }
        }
        
        # Suspicious behavior indicators
        self.suspicious_patterns = {
            'theft_indicators': [
                'pocket', 'bag', 'conceal', 'hide', 'reach', 'grab', 'take',
                'quick', 'sudden', 'rapid', 'sneak'
            ],
            'nervous_behavior': [
                'nervous', 'anxious', 'glancing', 'looking_around', 'avoiding',
                'fidget', 'tense', 'uncomfortable'
            ],
            'positioning': [
                'corner', 'edge', 'behind', 'away', 'hidden', 'blocked',
                'turning_away', 'facing_away'
            ]
        }
    
    def detect_theft_scenario(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential theft scenarios using rule-based analysis."""
        vlm_desc = analysis.get('vlm_description', '').lower()
        people_count = analysis.get('people_count', 0)
        objects_detected = analysis.get('objects_detected', [])
        
        theft_indicators = []
        risk_score = 0
        
        # Check for high-risk scenarios
        for scenario_name, scenario_config in self.high_risk_scenarios.items():
            if self._matches_scenario(vlm_desc, objects_detected, people_count, scenario_config):
                theft_indicators.append(scenario_name)
                risk_score += 10 * scenario_config['risk_multiplier']
        
        # Check for suspicious keywords in any available text
        all_text = f"{vlm_desc} {str(objects_detected)} {str(analysis.get('activity', ''))}"
        
        for category, keywords in self.suspicious_patterns.items():
            found_keywords = [kw for kw in keywords if kw in all_text]
            if found_keywords:
                theft_indicators.extend([f"{category}_{kw}" for kw in found_keywords])
                risk_score += len(found_keywords) * 3
        
        # Person count risk factor
        if people_count >= 4:
            theft_indicators.append('multiple_persons_high_risk')
            risk_score += 8
        elif people_count >= 2:
            theft_indicators.append('multiple_persons_medium_risk')
            risk_score += 4
        
        return {
            'theft_scenario_detected': len(theft_indicators) >= 2,
            'indicators': theft_indicators,
            'risk_score': risk_score,
            'risk_level': self._calculate_risk_level(risk_score),
            'confidence': min(0.95, 0.5 + (risk_score / 50))
        }
    
    def _matches_scenario(self, description: str, objects: List[str], people_count: int, 
                         scenario_config: Dict[str, Any]) -> bool:
        """Check if the scene matches a high-risk scenario."""
        # Check keywords in description
        keyword_match = any(kw in description for kw in scenario_config['keywords'])
        
        # Check objects
        object_match = any(obj in objects for obj in scenario_config['objects'])
        
        # Check people count
        people_match = people_count >= scenario_config['min_people']
        
        # Need at least 2 of 3 criteria
        matches = sum([keyword_match, object_match, people_match])
        return matches >= 2
    
    def _calculate_risk_level(self, risk_score: float) -> str:
        """Calculate risk level based on score."""
        if risk_score >= 20:
            return 'high'
        elif risk_score >= 10:
            return 'medium'
        else:
            return 'low'
    
    def generate_security_alert(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive security alert."""
        theft_detection = self.detect_theft_scenario(analysis)
        
        alert = {
            'alert_triggered': theft_detection['theft_scenario_detected'],
            'severity': theft_detection['risk_level'].upper(),
            'confidence': theft_detection['confidence'],
            'detection_method': 'rule_based_behavioral_analysis',
            'indicators': theft_detection['indicators'],
            'risk_score': theft_detection['risk_score']
        }
        
        # Generate specific alert reasoning
        if alert['alert_triggered']:
            alert['reasoning'] = self._generate_alert_reasoning(theft_detection, analysis)
            alert['recommended_action'] = self._generate_recommended_action(theft_detection)
        else:
            alert['reasoning'] = 'No significant behavioral threats detected'
            alert['recommended_action'] = 'Continue monitoring'
        
        return alert
    
    def _generate_alert_reasoning(self, theft_detection: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate detailed alert reasoning."""
        indicators = theft_detection['indicators']
        people_count = analysis.get('people_count', 0)
        
        reasoning_parts = []
        
        if 'retail_theft' in indicators:
            reasoning_parts.append("Retail environment with multiple people presents theft opportunity")
        
        if 'electronics_theft' in indicators:
            reasoning_parts.append("Electronics present with multiple persons increases theft risk")
        
        if any('theft_indicators' in ind for ind in indicators):
            reasoning_parts.append("Suspicious hand movements or concealment behaviors detected")
        
        if 'multiple_persons_high_risk' in indicators:
            reasoning_parts.append(f"High number of people ({people_count}) creates distraction opportunities")
        
        if not reasoning_parts:
            reasoning_parts.append("Multiple risk factors present requiring attention")
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_recommended_action(self, theft_detection: Dict[str, Any]) -> str:
        """Generate recommended security actions."""
        risk_level = theft_detection['risk_level']
        indicators = theft_detection['indicators']
        
        if risk_level == 'high':
            return "IMMEDIATE STAFF ATTENTION REQUIRED - Monitor all individuals closely, engage with customers, check merchandise security"
        elif risk_level == 'medium':
            return "Increase staff vigilance - Monitor interactions, verify customer needs, observe hand movements"
        else:
            return "Maintain standard observation - Be aware of customer interactions"

def detect_behavioral_threats_practical(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Main function for practical behavioral threat detection."""
    detector = PracticalBehavioralDetector()
    return detector.generate_security_alert(analysis)
