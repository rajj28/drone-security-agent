"""
analysis_agent.py - Analysis Agent with Reasoning

The Analysis Agent is responsible for:
- Deep video frame analysis
- Pattern recognition and behavioral analysis
- Threat detection and classification
- Temporal analysis across frames
- Generating comprehensive analysis reports
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
from pathlib import Path

from .base_agent import BaseAgent, AgentStatus, ReasoningStep
from langchain_core.pydantic_v1 import BaseModel, Field

@dataclass
class AnalysisResult:
    """Analysis result for a frame or video"""
    threat_level: str
    confidence: float
    patterns_detected: List[str]
    behaviors_observed: List[str]
    objects_detected: List[str]
    people_detected: int
    risk_score: float
    analysis_summary: str
    detailed_findings: Dict[str, Any]

class AnalysisAgent(BaseAgent):
    """Analysis Agent with comprehensive reasoning capabilities"""
    
    def __init__(self, openai_api_key: str = None):
        super().__init__("analysis_agent", openai_api_key)
        
        # Analysis-specific metrics
        self.analysis_metrics = {
            'frames_analyzed': 0,
            'threats_detected': 0,
            'patterns_identified': 0,
            'behaviors_classified': 0,
            'avg_risk_score': 0.0,
            'high_confidence_analyses': 0,
            'temporal_patterns_found': 0
        }
        
        # Analysis parameters
        self.analysis_params = {
            'threat_keywords': [
                'theft', 'stealing', 'shoplifting', 'concealing', 'hiding',
                'grabbing', 'taking', 'suspicious', 'unusual', 'nervous',
                'looking around', 'scanning', 'monitoring', 'distracting'
            ],
            'behavior_patterns': [
                'hand_in_pocket', 'looking_around', 'quick_movements',
                'avoiding_eye_contact', 'group_coordination', 'distraction'
            ],
            'risk_weights': {
                'high_threat': 0.8,
                'medium_threat': 0.5,
                'low_threat': 0.2,
                'no_threat': 0.0
            }
        }
    
    def get_system_prompt(self) -> str:
        """Get system prompt for analysis agent"""
        return """
        You are a Video Analysis Agent specializing in security threat detection.
        
        Your responsibilities:
        1. Analyze video frames for security threats and suspicious behavior
        2. Identify patterns and behavioral indicators
        3. Detect objects, people, and activities of interest
        4. Assess risk levels and provide confidence scores
        5. Generate detailed analysis with clear reasoning
        6. Perform temporal analysis across multiple frames
        
        You should be thorough, objective, and provide detailed explanations.
        Always explain your reasoning step by step and cite specific evidence.
        """
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data for comprehensive analysis"""
        start_time = datetime.utcnow()
        
        try:
            self.update_status(AgentStatus.PROCESSING, "Comprehensive video analysis")
            
            # Extract data
            frame_analyses = input_data.get('frame_analyses', [])
            video_metadata = input_data.get('video_metadata', {})
            analysis_config = input_data.get('analysis_config', {})
            
            # Step 1: Individual frame analysis
            frame_results = await self._analyze_individual_frames(frame_analyses)
            
            # Step 2: Pattern recognition across frames
            pattern_results = await self._recognize_patterns(frame_analyses, frame_results)
            
            # Step 3: Behavioral analysis
            behavior_results = await self._analyze_behaviors(frame_analyses, frame_results)
            
            # Step 4: Temporal analysis
            temporal_results = await self._perform_temporal_analysis(frame_analyses, frame_results)
            
            # Step 5: Risk assessment
            risk_assessment = await self._assess_risk(frame_results, pattern_results, behavior_results)
            
            # Step 6: Generate comprehensive report
            comprehensive_report = await self._generate_comprehensive_report({
                'frame_results': frame_results,
                'pattern_results': pattern_results,
                'behavior_results': behavior_results,
                'temporal_results': temporal_results,
                'risk_assessment': risk_assessment,
                'video_metadata': video_metadata
            })
            
            # Calculate duration
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Update metrics
            self.analysis_metrics['frames_analyzed'] += len(frame_analyses)
            self.analysis_metrics['threats_detected'] += len([r for r in frame_results if r.get('threat_level') in ['high', 'medium']])
            self.analysis_metrics['patterns_identified'] += len(pattern_results.get('patterns', []))
            self.analysis_metrics['behaviors_classified'] += len(behavior_results.get('behaviors', []))
            
            if risk_assessment.get('overall_risk_score', 0) > 0.7:
                self.analysis_metrics['high_confidence_analyses'] += 1
            
            # Update average risk score
            total_analyses = self.analysis_metrics['frames_analyzed']
            if total_analyses > 0:
                self.analysis_metrics['avg_risk_score'] = (
                    (self.analysis_metrics['avg_risk_score'] * (total_analyses - len(frame_analyses)) + 
                     sum(r.get('risk_score', 0) for r in frame_results)) / total_analyses
                )
            
            # Add final reasoning step
            self.add_reasoning_step(
                step_type="comprehensive_analysis",
                description="Complete video analysis with all components",
                input_data=input_data,
                reasoning=f"Completed comprehensive analysis of {len(frame_analyses)} frames. Overall risk: {risk_assessment.get('overall_risk_score', 0):.3f}",
                output_data=comprehensive_report,
                confidence=risk_assessment.get('overall_risk_score', 0),
                duration_ms=duration_ms,
                metadata={'analysis_type': 'comprehensive', 'frames_processed': len(frame_analyses)}
            )
            
            self.update_status(AgentStatus.COMPLETED)
            self.metrics['tasks_completed'] += 1
            
            return {
                'agent_id': self.agent_id,
                'agent_name': self.agent_name,
                'analysis_result': comprehensive_report,
                'reasoning_chain': self.get_reasoning_chain(),
                'metrics': self.get_metrics(),
                'analysis_metrics': self.analysis_metrics,
                'processing_time_ms': duration_ms
            }
            
        except Exception as e:
            self.update_status(AgentStatus.ERROR)
            self.metrics['tasks_failed'] += 1
            logger.error(f"Analysis processing failed: {e}")
            raise
    
    async def _analyze_individual_frames(self, frame_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze individual frames in detail"""
        
        start_time = datetime.utcnow()
        frame_results = []
        
        for i, frame in enumerate(frame_analyses):
            frame_start = datetime.utcnow()
            
            frame_id = frame.get('frame_id', f'frame_{i}')
            analysis_result = frame.get('analysis_result', {})
            
            # Extract basic information
            confidence = frame.get('confidence', 0.0)
            threat_level = frame.get('threat_level', 'low')
            
            # Analyze GPT-4 enhanced analysis if available
            enhanced_analysis = ""
            if 'gpt4o_enhanced' in analysis_result:
                gpt4o_result = analysis_result['gpt4o_enhanced']
                if 'enhanced_analysis' in gpt4o_result:
                    enhanced_analysis = str(gpt4o_result['enhanced_analysis'])
            
            # Detect threat keywords
            threat_keywords_found = []
            for keyword in self.analysis_params['threat_keywords']:
                if keyword in enhanced_analysis.lower():
                    threat_keywords_found.append(keyword)
            
            # Detect behavior patterns
            behavior_patterns_found = []
            for pattern in self.analysis_params['behavior_patterns']:
                if pattern.replace('_', ' ') in enhanced_analysis.lower():
                    behavior_patterns_found.append(pattern)
            
            # Calculate risk score
            risk_score = self._calculate_frame_risk_score(
                threat_level, confidence, threat_keywords_found, behavior_patterns_found
            )
            
            # Generate frame summary
            frame_summary = await self._generate_frame_summary(
                frame_id, enhanced_analysis, threat_keywords_found, behavior_patterns_found
            )
            
            frame_result = {
                'frame_id': frame_id,
                'frame_number': frame.get('frame_number', i + 1),
                'threat_level': threat_level,
                'confidence': confidence,
                'risk_score': risk_score,
                'threat_keywords_found': threat_keywords_found,
                'behavior_patterns_found': behavior_patterns_found,
                'people_detected': analysis_result.get('people_detected', 0),
                'objects_detected': analysis_result.get('objects_detected', []),
                'enhanced_analysis': enhanced_analysis,
                'frame_summary': frame_summary,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            frame_results.append(frame_result)
            
            # Add reasoning step for this frame
            frame_duration_ms = int((datetime.utcnow() - frame_start).total_seconds() * 1000)
            
            self.add_reasoning_step(
                step_type="frame_analysis",
                description=f"Analyzing frame {frame_id}",
                input_data={'frame_id': frame_id, 'confidence': confidence},
                reasoning=f"Frame {frame_id}: {threat_level} threat (confidence: {confidence:.3f}, risk: {risk_score:.3f}). Found {len(threat_keywords_found)} threat keywords and {len(behavior_patterns_found)} behavior patterns.",
                output_data=frame_result,
                confidence=confidence,
                duration_ms=frame_duration_ms,
                metadata={'frame_number': i + 1, 'risk_score': risk_score}
            )
        
        # Add summary reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="frames_summary",
            description="Summary of individual frame analyses",
            input_data={'total_frames': len(frame_analyses)},
            reasoning=f"Analyzed {len(frame_results)} frames. Average risk score: {sum(r['risk_score'] for r in frame_results) / len(frame_results):.3f}",
            output_data={'frame_count': len(frame_results), 'avg_risk_score': sum(r['risk_score'] for r in frame_results) / len(frame_results)},
            confidence=0.9,
            duration_ms=duration_ms
        )
        
        return frame_results
    
    async def _recognize_patterns(self, frame_analyses: List[Dict[str, Any]], frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recognize patterns across multiple frames"""
        
        start_time = datetime.utcnow()
        
        # Collect all threat keywords and behaviors
        all_threat_keywords = []
        all_behavior_patterns = []
        
        for result in frame_results:
            all_threat_keywords.extend(result['threat_keywords_found'])
            all_behavior_patterns.extend(result['behavior_patterns_found'])
        
        # Count frequency of patterns
        threat_keyword_counts = {}
        for keyword in all_threat_keywords:
            threat_keyword_counts[keyword] = threat_keyword_counts.get(keyword, 0) + 1
        
        behavior_pattern_counts = {}
        for pattern in all_behavior_patterns:
            behavior_pattern_counts[pattern] = behavior_pattern_counts.get(pattern, 0) + 1
        
        # Identify significant patterns (appearing in multiple frames)
        significant_threat_patterns = {
            k: v for k, v in threat_keyword_counts.items() if v >= 2
        }
        
        significant_behavior_patterns = {
            k: v for k, v in behavior_pattern_counts.items() if v >= 2
        }
        
        # Analyze temporal patterns
        temporal_patterns = []
        for i, result in enumerate(frame_results):
            if result['threat_level'] in ['high', 'medium']:
                temporal_patterns.append({
                    'frame_number': result['frame_number'],
                    'threat_level': result['threat_level'],
                    'risk_score': result['risk_score'],
                    'patterns': result['threat_keywords_found'] + result['behavior_patterns_found']
                })
        
        # Generate pattern insights
        pattern_insights = await self._generate_pattern_insights(
            significant_threat_patterns, significant_behavior_patterns, temporal_patterns
        )
        
        # Update metrics
        self.analysis_metrics['temporal_patterns_found'] += len(temporal_patterns)
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="pattern_recognition",
            description="Recognizing patterns across frames",
            input_data={'frame_count': len(frame_results)},
            reasoning=f"Identified {len(significant_threat_patterns)} significant threat patterns and {len(significant_behavior_patterns)} behavior patterns across {len(frame_results)} frames",
            output_data={
                'significant_threat_patterns': significant_threat_patterns,
                'significant_behavior_patterns': significant_behavior_patterns,
                'temporal_patterns': temporal_patterns,
                'pattern_insights': pattern_insights
            },
            confidence=0.85,
            duration_ms=duration_ms
        )
        
        return {
            'patterns': list(significant_threat_patterns.keys()) + list(significant_behavior_patterns.keys()),
            'threat_patterns': significant_threat_patterns,
            'behavior_patterns': significant_behavior_patterns,
            'temporal_patterns': temporal_patterns,
            'pattern_insights': pattern_insights
        }
    
    async def _analyze_behaviors(self, frame_analyses: List[Dict[str, Any]], frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze behaviors across frames"""
        
        start_time = datetime.utcnow()
        
        # Classify behaviors
        behavior_categories = {
            'suspicious_movements': [],
            'concealment_actions': [],
            'surveillance_behavior': [],
            'coordination_patterns': [],
            'normal_behavior': []
        }
        
        for result in frame_results:
            behaviors = result['behavior_patterns_found']
            
            for behavior in behaviors:
                if behavior in ['quick_movements', 'suspicious']:
                    behavior_categories['suspicious_movements'].append(result['frame_id'])
                elif behavior in ['hand_in_pocket', 'concealing', 'hiding']:
                    behavior_categories['concealment_actions'].append(result['frame_id'])
                elif behavior in ['looking_around', 'scanning', 'monitoring']:
                    behavior_categories['surveillance_behavior'].append(result['frame_id'])
                elif behavior in ['group_coordination']:
                    behavior_categories['coordination_patterns'].append(result['frame_id'])
                else:
                    behavior_categories['normal_behavior'].append(result['frame_id'])
        
        # Analyze behavior sequences
        behavior_sequences = []
        current_sequence = []
        
        for i, result in enumerate(frame_results):
            frame_behaviors = result['behavior_patterns_found']
            
            if frame_behaviors:
                if not current_sequence:
                    current_sequence = [result['frame_id']]
                else:
                    # Check if this is a continuation
                    prev_frame = frame_results[i-1]
                    if prev_frame['behavior_patterns_found']:
                        current_sequence.append(result['frame_id'])
                    else:
                        if len(current_sequence) > 1:
                            behavior_sequences.append(current_sequence)
                        current_sequence = [result['frame_id']]
            else:
                if len(current_sequence) > 1:
                    behavior_sequences.append(current_sequence)
                current_sequence = []
        
        # Generate behavior analysis
        behavior_analysis = await self._generate_behavior_analysis(
            behavior_categories, behavior_sequences, frame_results
        )
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="behavior_analysis",
            description="Analyzing behaviors across frames",
            input_data={'frame_count': len(frame_results)},
            reasoning=f"Classified behaviors into {len(behavior_categories)} categories and identified {len(behavior_sequences)} behavior sequences",
            output_data={
                'behavior_categories': behavior_categories,
                'behavior_sequences': behavior_sequences,
                'behavior_analysis': behavior_analysis
            },
            confidence=0.8,
            duration_ms=duration_ms
        )
        
        return {
            'behaviors': list(behavior_categories.keys()),
            'behavior_categories': behavior_categories,
            'behavior_sequences': behavior_sequences,
            'behavior_analysis': behavior_analysis
        }
    
    async def _perform_temporal_analysis(self, frame_analyses: List[Dict[str, Any]], frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform temporal analysis across frames"""
        
        start_time = datetime.utcnow()
        
        # Analyze threat progression
        threat_progression = []
        for result in frame_results:
            threat_progression.append({
                'frame_number': result['frame_number'],
                'threat_level': result['threat_level'],
                'risk_score': result['risk_score'],
                'confidence': result['confidence']
            })
        
        # Identify threat escalation points
        escalation_points = []
        for i in range(1, len(threat_progression)):
            prev = threat_progression[i-1]
            curr = threat_progression[i]
            
            # Check for escalation
            if (prev['threat_level'] == 'low' and curr['threat_level'] in ['medium', 'high']) or \
               (prev['threat_level'] == 'medium' and curr['threat_level'] == 'high'):
                escalation_points.append({
                    'frame_number': curr['frame_number'],
                    'from_threat': prev['threat_level'],
                    'to_threat': curr['threat_level'],
                    'risk_increase': curr['risk_score'] - prev['risk_score']
                })
        
        # Analyze risk trends
        risk_scores = [r['risk_score'] for r in frame_results]
        risk_trend = 'stable'
        if len(risk_scores) > 1:
            if risk_scores[-1] > risk_scores[0]:
                risk_trend = 'increasing'
            elif risk_scores[-1] < risk_scores[0]:
                risk_trend = 'decreasing'
        
        # Generate temporal insights
        temporal_insights = await self._generate_temporal_insights(
            threat_progression, escalation_points, risk_trend
        )
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="temporal_analysis",
            description="Performing temporal analysis across frames",
            input_data={'frame_count': len(frame_results)},
            reasoning=f"Analyzed threat progression across {len(frame_results)} frames. Found {len(escalation_points)} escalation points. Risk trend: {risk_trend}",
            output_data={
                'threat_progression': threat_progression,
                'escalation_points': escalation_points,
                'risk_trend': risk_trend,
                'temporal_insights': temporal_insights
            },
            confidence=0.85,
            duration_ms=duration_ms
        )
        
        return {
            'threat_progression': threat_progression,
            'escalation_points': escalation_points,
            'risk_trend': risk_trend,
            'temporal_insights': temporal_insights
        }
    
    async def _assess_risk(self, frame_results: List[Dict[str, Any]], pattern_results: Dict[str, Any], behavior_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk"""
        
        start_time = datetime.utcnow()
        
        # Calculate risk components
        frame_risk_scores = [r['risk_score'] for r in frame_results]
        avg_frame_risk = sum(frame_risk_scores) / len(frame_risk_scores) if frame_risk_scores else 0.0
        
        max_frame_risk = max(frame_risk_scores) if frame_risk_scores else 0.0
        
        # Pattern risk
        pattern_risk = min(len(pattern_results.get('patterns', [])) * 0.1, 1.0)
        
        # Behavior risk
        suspicious_behaviors = len(behavior_results.get('behavior_categories', {}).get('suspicious_movements', []))
        behavior_risk = min(suspicious_behaviors * 0.15, 1.0)
        
        # Overall risk score
        overall_risk_score = (
            avg_frame_risk * 0.4 +
            max_frame_risk * 0.3 +
            pattern_risk * 0.2 +
            behavior_risk * 0.1
        )
        
        # Risk classification
        if overall_risk_score >= 0.8:
            risk_level = 'critical'
        elif overall_risk_score >= 0.6:
            risk_level = 'high'
        elif overall_risk_score >= 0.4:
            risk_level = 'medium'
        elif overall_risk_score >= 0.2:
            risk_level = 'low'
        else:
            risk_level = 'minimal'
        
        # Generate risk assessment
        risk_assessment = await self._generate_risk_assessment(
            overall_risk_score, risk_level, frame_results, pattern_results, behavior_results
        )
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="risk_assessment",
            description="Assessing overall security risk",
            input_data={'frame_count': len(frame_results)},
            reasoning=f"Calculated overall risk score {overall_risk_score:.3f} (risk level: {risk_level}) from frame risk ({avg_frame_risk:.3f}), pattern risk ({pattern_risk:.3f}), and behavior risk ({behavior_risk:.3f})",
            output_data={
                'overall_risk_score': overall_risk_score,
                'risk_level': risk_level,
                'risk_components': {
                    'avg_frame_risk': avg_frame_risk,
                    'max_frame_risk': max_frame_risk,
                    'pattern_risk': pattern_risk,
                    'behavior_risk': behavior_risk
                },
                'risk_assessment': risk_assessment
            },
            confidence=overall_risk_score,
            duration_ms=duration_ms
        )
        
        return {
            'overall_risk_score': overall_risk_score,
            'risk_level': risk_level,
            'risk_components': {
                'avg_frame_risk': avg_frame_risk,
                'max_frame_risk': max_frame_risk,
                'pattern_risk': pattern_risk,
                'behavior_risk': behavior_risk
            },
            'risk_assessment': risk_assessment
        }
    
    async def _generate_comprehensive_report(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        
        start_time = datetime.utcnow()
        
        frame_results = analysis_data['frame_results']
        pattern_results = analysis_data['pattern_results']
        behavior_results = analysis_data['behavior_results']
        temporal_results = analysis_data['temporal_results']
        risk_assessment = analysis_data['risk_assessment']
        video_metadata = analysis_data['video_metadata']
        
        # Executive summary
        executive_summary = await self._generate_executive_summary(
            frame_results, pattern_results, behavior_results, temporal_results, risk_assessment
        )
        
        # Detailed findings
        detailed_findings = {
            'threat_analysis': {
                'total_frames': len(frame_results),
                'threat_frames': len([r for r in frame_results if r['threat_level'] in ['high', 'medium']]),
                'high_risk_frames': len([r for r in frame_results if r['risk_score'] > 0.7]),
                'avg_risk_score': sum(r['risk_score'] for r in frame_results) / len(frame_results) if frame_results else 0.0
            },
            'pattern_analysis': {
                'patterns_found': len(pattern_results.get('patterns', [])),
                'significant_patterns': len(pattern_results.get('threat_patterns', {})) + len(pattern_results.get('behavior_patterns', {})),
                'temporal_patterns': len(pattern_results.get('temporal_patterns', []))
            },
            'behavior_analysis': {
                'behaviors_classified': len(behavior_results.get('behaviors', [])),
                'suspicious_behaviors': len(behavior_results.get('behavior_categories', {}).get('suspicious_movements', [])),
                'behavior_sequences': len(behavior_results.get('behavior_sequences', []))
            },
            'temporal_analysis': {
                'risk_trend': temporal_results.get('risk_trend', 'stable'),
                'escalation_points': len(temporal_results.get('escalation_points', [])),
                'threat_progression': temporal_results.get('threat_progression', [])
            }
        }
        
        # Recommendations
        recommendations = await self._generate_recommendations(
            risk_assessment, frame_results, pattern_results, behavior_results
        )
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="report_generation",
            description="Generating comprehensive analysis report",
            input_data={'analysis_components': list(analysis_data.keys())},
            reasoning=f"Generated comprehensive report with executive summary, detailed findings, and recommendations. Overall risk: {risk_assessment['overall_risk_score']:.3f}",
            output_data={
                'executive_summary': executive_summary,
                'detailed_findings': detailed_findings,
                'recommendations': recommendations,
                'video_metadata': video_metadata,
                'analysis_timestamp': datetime.utcnow().isoformat()
            },
            confidence=risk_assessment['overall_risk_score'],
            duration_ms=duration_ms
        )
        
        return {
            'executive_summary': executive_summary,
            'detailed_findings': detailed_findings,
            'recommendations': recommendations,
            'video_metadata': video_metadata,
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'risk_assessment': risk_assessment,
            'analysis_components': {
                'frame_results': len(frame_results),
                'patterns_found': len(pattern_results.get('patterns', [])),
                'behaviors_classified': len(behavior_results.get('behaviors', [])),
                'temporal_patterns': len(temporal_results.get('temporal_patterns', []))
            }
        }
    
    def _calculate_frame_risk_score(self, threat_level: str, confidence: float, threat_keywords: List[str], behavior_patterns: List[str]) -> float:
        """Calculate risk score for a frame"""
        
        base_risk = self.analysis_params['risk_weights'].get(threat_level, 0.0)
        
        # Adjust for confidence
        confidence_adjustment = confidence
        
        # Adjust for threat keywords
        keyword_adjustment = min(len(threat_keywords) * 0.1, 0.3)
        
        # Adjust for behavior patterns
        behavior_adjustment = min(len(behavior_patterns) * 0.15, 0.4)
        
        # Calculate final risk score
        risk_score = base_risk + (confidence_adjustment * 0.3) + keyword_adjustment + behavior_adjustment
        
        return min(risk_score, 1.0)
    
    async def _generate_frame_summary(self, frame_id: str, enhanced_analysis: str, threat_keywords: List[str], behavior_patterns: List[str]) -> str:
        """Generate summary for a frame"""
        
        prompt = f"""
        Analyze this frame analysis and provide a concise summary:
        
        Frame ID: {frame_id}
        Enhanced Analysis: {enhanced_analysis}
        Threat Keywords Found: {threat_keywords}
        Behavior Patterns: {behavior_patterns}
        
        Provide a 2-3 sentence summary focusing on key findings and security implications.
        """
        
        try:
            response = await self.call_llm([prompt])
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate frame summary: {e}")
            return f"Frame {frame_id} analysis completed with {len(threat_keywords)} threat indicators and {len(behavior_patterns)} behavior patterns."
    
    async def _generate_pattern_insights(self, threat_patterns: Dict, behavior_patterns: Dict, temporal_patterns: List) -> str:
        """Generate insights about patterns"""
        
        prompt = f"""
        Analyze these patterns and provide security insights:
        
        Threat Patterns: {threat_patterns}
        Behavior Patterns: {behavior_patterns}
        Temporal Patterns: {temporal_patterns}
        
        What security insights can be derived from these patterns?
        """
        
        try:
            response = await self.call_llm([prompt])
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate pattern insights: {e}")
            return "Pattern analysis completed with multiple indicators detected."
    
    async def _generate_behavior_analysis(self, behavior_categories: Dict, behavior_sequences: List, frame_results: List) -> str:
        """Generate behavior analysis"""
        
        prompt = f"""
        Analyze these behaviors and provide security assessment:
        
        Behavior Categories: {behavior_categories}
        Behavior Sequences: {behavior_sequences}
        
        What do these behaviors indicate from a security perspective?
        """
        
        try:
            response = await self.call_llm([prompt])
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate behavior analysis: {e}")
            return "Behavior analysis completed with suspicious activities detected."
    
    async def _generate_temporal_insights(self, threat_progression: List, escalation_points: List, risk_trend: str) -> str:
        """Generate temporal insights"""
        
        prompt = f"""
        Analyze this temporal data and provide security insights:
        
        Threat Progression: {threat_progression}
        Escalation Points: {escalation_points}
        Risk Trend: {risk_trend}
        
        What security insights can be derived from this temporal analysis?
        """
        
        try:
            response = await self.call_llm([prompt])
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate temporal insights: {e}")
            return f"Temporal analysis shows {risk_trend} risk trend with {len(escalation_points)} escalation points."
    
    async def _generate_risk_assessment(self, overall_risk: float, risk_level: str, frame_results: List, pattern_results: Dict, behavior_results: Dict) -> str:
        """Generate risk assessment"""
        
        prompt = f"""
        Provide a comprehensive risk assessment based on:
        
        Overall Risk Score: {overall_risk:.3f}
        Risk Level: {risk_level}
        Frame Results Summary: {len(frame_results)} frames analyzed
        Patterns Found: {pattern_results.get('patterns', [])}
        Behaviors Detected: {behavior_results.get('behaviors', [])}
        
        Provide a detailed risk assessment with specific concerns and recommendations.
        """
        
        try:
            response = await self.call_llm([prompt])
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate risk assessment: {e}")
            return f"Risk assessment: {risk_level} risk level detected with score {overall_risk:.3f}."
    
    async def _generate_executive_summary(self, frame_results: List, pattern_results: Dict, behavior_results: Dict, temporal_results: Dict, risk_assessment: Dict) -> str:
        """Generate executive summary"""
        
        prompt = f"""
        Generate an executive summary for security analysis:
        
        Risk Level: {risk_assessment['risk_level']}
        Risk Score: {risk_assessment['overall_risk_score']:.3f}
        Frames Analyzed: {len(frame_results)}
        Threat Frames: {len([r for r in frame_results if r['threat_level'] in ['high', 'medium']])}
        Patterns Found: {len(pattern_results.get('patterns', []))}
        Risk Trend: {temporal_results.get('risk_trend', 'stable')}
        
        Provide a concise 3-4 sentence executive summary for security personnel.
        """
        
        try:
            response = await self.call_llm([prompt])
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return f"Analysis completed with {risk_assessment['risk_level']} risk level. Immediate attention recommended."
    
    async def _generate_recommendations(self, risk_assessment: Dict, frame_results: List, pattern_results: Dict, behavior_results: Dict) -> List[str]:
        """Generate security recommendations"""
        
        prompt = f"""
        Based on this analysis, provide 3-5 specific security recommendations:
        
        Risk Level: {risk_assessment['risk_level']}
        Risk Score: {risk_assessment['overall_risk_score']:.3f}
        Patterns: {pattern_results.get('patterns', [])}
        Behaviors: {behavior_results.get('behaviors', [])}
        
        Provide actionable recommendations as a numbered list.
        """
        
        try:
            response = await self.call_llm([prompt])
            # Parse numbered list
            recommendations = []
            for line in response.strip().split('\n'):
                if line.strip() and (line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '-', '*'))):
                    recommendations.append(line.strip())
            return recommendations
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Review security footage", "Increase monitoring", "Staff training recommended"]
    
    def get_analysis_metrics(self) -> Dict[str, Any]:
        """Get analysis-specific metrics"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'analysis_metrics': self.analysis_metrics,
            'analysis_params': self.analysis_params,
            'general_metrics': self.get_metrics()
        }
