"""
qa_agent.py - Quality Assurance Agent with Reasoning

The QA Agent is responsible for:
- Validating analysis results
- Checking for false positives/negatives
- Ensuring data quality
- Generating quality reports
- Providing confidence scores
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re

from .base_agent import BaseAgent, AgentStatus, ReasoningStep
from langchain_core.pydantic_v1 import BaseModel, Field

@dataclass
class QAResult:
    """Quality assurance result"""
    is_valid: bool
    confidence_score: float
    issues_found: List[str]
    recommendations: List[str]
    quality_score: float
    validation_details: Dict[str, Any]

class QAAgent(BaseAgent):
    """Quality Assurance Agent with comprehensive reasoning"""
    
    def __init__(self, openai_api_key: str = None):
        super().__init__("qa_agent", openai_api_key)
        
        # QA-specific metrics
        self.qa_metrics = {
            'validations_performed': 0,
            'issues_detected': 0,
            'false_positives_identified': 0,
            'false_negatives_identified': 0,
            'avg_quality_score': 0.0,
            'high_confidence_validations': 0
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_confidence': 0.7,
            'max_false_positive_rate': 0.1,
            'min_data_completeness': 0.9,
            'max_processing_time_ms': 30000
        }
    
    def get_system_prompt(self) -> str:
        """Get system prompt for QA agent"""
        return """
        You are a Quality Assurance Agent for a drone security analysis system.
        
        Your responsibilities:
        1. Validate analysis results for accuracy and consistency
        2. Identify potential false positives and false negatives
        3. Check data quality and completeness
        4. Assess confidence levels and provide recommendations
        5. Generate quality reports with detailed reasoning
        
        You should be thorough, critical, and provide specific feedback.
        Always explain your reasoning step by step.
        """
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data for quality assurance"""
        start_time = datetime.utcnow()
        
        try:
            self.update_status(AgentStatus.PROCESSING, "Quality assurance validation")
            
            # Extract data
            analysis_results = input_data.get('analysis_results', {})
            video_metadata = input_data.get('video_metadata', {})
            frame_analyses = input_data.get('frame_analyses', [])
            
            # Step 1: Data completeness check
            completeness_result = await self._check_data_completeness(
                analysis_results, video_metadata, frame_analyses
            )
            
            # Step 2: Confidence validation
            confidence_result = await self._validate_confidence_levels(frame_analyses)
            
            # Step 3: False positive detection
            false_positive_result = await self._detect_false_positives(frame_analyses)
            
            # Step 4: False negative detection
            false_negative_result = await self._detect_false_negatives(frame_analyses)
            
            # Step 5: Consistency check
            consistency_result = await self._check_consistency(frame_analyses)
            
            # Step 6: Generate overall quality assessment
            quality_assessment = await self._generate_quality_assessment({
                'completeness': completeness_result,
                'confidence': confidence_result,
                'false_positives': false_positive_result,
                'false_negatives': false_negative_result,
                'consistency': consistency_result
            })
            
            # Calculate duration
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Update metrics
            self.qa_metrics['validations_performed'] += 1
            self.qa_metrics['issues_detected'] += len(quality_assessment['issues_found'])
            self.qa_metrics['avg_quality_score'] = (
                (self.qa_metrics['avg_quality_score'] * (self.qa_metrics['validations_performed'] - 1) + 
                 quality_assessment['quality_score']) / self.qa_metrics['validations_performed']
            )
            
            if quality_assessment['confidence_score'] > 0.8:
                self.qa_metrics['high_confidence_validations'] += 1
            
            # Add final reasoning step
            self.add_reasoning_step(
                step_type="final_assessment",
                description="Complete quality assurance validation",
                input_data=input_data,
                reasoning=f"Completed comprehensive QA check with quality score {quality_assessment['quality_score']:.2f}",
                output_data=quality_assessment,
                confidence=quality_assessment['confidence_score'],
                duration_ms=duration_ms,
                metadata={'validation_type': 'comprehensive_qa'}
            )
            
            self.update_status(AgentStatus.COMPLETED)
            self.metrics['tasks_completed'] += 1
            
            return {
                'agent_id': self.agent_id,
                'agent_name': self.agent_name,
                'qa_result': quality_assessment,
                'reasoning_chain': self.get_reasoning_chain(),
                'metrics': self.get_metrics(),
                'qa_metrics': self.qa_metrics,
                'processing_time_ms': duration_ms
            }
            
        except Exception as e:
            self.update_status(AgentStatus.ERROR)
            self.metrics['tasks_failed'] += 1
            logger.error(f"QA processing failed: {e}")
            raise
    
    async def _check_data_completeness(
        self, 
        analysis_results: Dict[str, Any], 
        video_metadata: Dict[str, Any], 
        frame_analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check data completeness and quality"""
        
        start_time = datetime.utcnow()
        
        # Check required fields
        required_video_fields = ['video_id', 'filename', 'duration', 'fps', 'resolution']
        missing_video_fields = [field for field in required_video_fields if field not in video_metadata]
        
        required_analysis_fields = ['overall_threat_level', 'confidence', 'people_detected']
        missing_analysis_fields = [field for field in required_analysis_fields if field not in analysis_results]
        
        # Check frame analysis completeness
        total_frames = len(frame_analyses)
        frames_with_analysis = len([f for f in frame_analyses if f.get('analysis_result')])
        frames_with_threats = len([f for f in frame_analyses if f.get('threat_level', '').lower() in ['high', 'medium']])
        
        completeness_score = 1.0
        issues = []
        
        if missing_video_fields:
            completeness_score -= 0.2 * len(missing_video_fields)
            issues.append(f"Missing video metadata: {missing_video_fields}")
        
        if missing_analysis_fields:
            completeness_score -= 0.15 * len(missing_analysis_fields)
            issues.append(f"Missing analysis fields: {missing_analysis_fields}")
        
        if frames_with_analysis < total_frames:
            completeness_score -= 0.3 * (1 - frames_with_analysis / total_frames)
            issues.append(f"Only {frames_with_analysis}/{total_frames} frames have analysis")
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="data_completeness",
            description="Checking data completeness and quality",
            input_data={
                'video_fields': list(video_metadata.keys()),
                'analysis_fields': list(analysis_results.keys()),
                'total_frames': total_frames
            },
            reasoning=f"Checked {len(required_video_fields)} video fields and {len(required_analysis_fields)} analysis fields. Found {len(missing_video_fields)} missing video fields and {len(missing_analysis_fields)} missing analysis fields. Frame analysis completeness: {frames_with_analysis}/{total_frames}",
            output_data={
                'completeness_score': completeness_score,
                'issues': issues,
                'missing_video_fields': missing_video_fields,
                'missing_analysis_fields': missing_analysis_fields,
                'frames_with_analysis': frames_with_analysis,
                'total_frames': total_frames
            },
            confidence=completeness_score,
            duration_ms=duration_ms
        )
        
        return {
            'completeness_score': max(0, completeness_score),
            'issues': issues,
            'missing_video_fields': missing_video_fields,
            'missing_analysis_fields': missing_analysis_fields,
            'frames_with_analysis': frames_with_analysis,
            'total_frames': total_frames
        }
    
    async def _validate_confidence_levels(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate confidence levels in frame analyses"""
        
        start_time = datetime.utcnow()
        
        confidence_scores = []
        low_confidence_frames = []
        
        for frame in frame_analyses:
            confidence = frame.get('confidence', 0.0)
            confidence_scores.append(confidence)
            
            if confidence < self.quality_thresholds['min_confidence']:
                low_confidence_frames.append({
                    'frame_id': frame.get('frame_id'),
                    'confidence': confidence,
                    'threat_level': frame.get('threat_level')
                })
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        confidence_score = min(avg_confidence / self.quality_thresholds['min_confidence'], 1.0)
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="confidence_validation",
            description="Validating confidence levels across frames",
            input_data={'frame_count': len(frame_analyses)},
            reasoning=f"Analyzed {len(frame_analyses)} frames. Average confidence: {avg_confidence:.3f}. Found {len(low_confidence_frames)} frames below threshold ({self.quality_thresholds['min_confidence']})",
            output_data={
                'avg_confidence': avg_confidence,
                'confidence_score': confidence_score,
                'low_confidence_frames': low_confidence_frames,
                'threshold': self.quality_thresholds['min_confidence']
            },
            confidence=confidence_score,
            duration_ms=duration_ms
        )
        
        return {
            'avg_confidence': avg_confidence,
            'confidence_score': confidence_score,
            'low_confidence_frames': low_confidence_frames,
            'threshold': self.quality_thresholds['min_confidence']
        }
    
    async def _detect_false_positives(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect potential false positives in threat detection"""
        
        start_time = datetime.utcnow()
        
        threat_frames = [f for f in frame_analyses if f.get('threat_level', '').lower() in ['high', 'medium']]
        potential_false_positives = []
        
        for frame in threat_frames:
            frame_id = frame.get('frame_id')
            confidence = frame.get('confidence', 0.0)
            analysis_result = frame.get('analysis_result', {})
            
            # Check for indicators of false positives
            false_positive_indicators = []
            
            # Low confidence threat
            if confidence < 0.6:
                false_positive_indicators.append("Low confidence threat detection")
            
            # No people detected but threat flagged
            people_detected = analysis_result.get('people_detected', 0)
            if people_detected == 0:
                false_positive_indicators.append("Threat detected with no people identified")
            
            # Check analysis text for inconsistencies
            if 'gpt4o_enhanced' in analysis_result:
                gpt4o_result = analysis_result['gpt4o_enhanced']
                if 'enhanced_analysis' in gpt4o_result:
                    analysis_text = str(gpt4o_result['enhanced_analysis']).lower()
                    
                    # Look for contradictory statements
                    if 'normal' in analysis_text and 'threat' in analysis_text:
                        false_positive_indicators.append("Contradictory analysis (normal vs threat)")
                    
                    # Look for uncertainty indicators
                    uncertainty_words = ['unclear', 'uncertain', 'maybe', 'possibly', 'might be']
                    if any(word in analysis_text for word in uncertainty_words):
                        false_positive_indicators.append("Analysis contains uncertainty indicators")
            
            # If multiple indicators found, flag as potential false positive
            if len(false_positive_indicators) >= 2:
                potential_false_positives.append({
                    'frame_id': frame_id,
                    'confidence': confidence,
                    'threat_level': frame.get('threat_level'),
                    'indicators': false_positive_indicators
                })
        
        false_positive_rate = len(potential_false_positives) / len(threat_frames) if threat_frames else 0.0
        false_positive_score = 1.0 - min(false_positive_rate / self.quality_thresholds['max_false_positive_rate'], 1.0)
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="false_positive_detection",
            description="Detecting potential false positives in threat detection",
            input_data={'threat_frames': len(threat_frames)},
            reasoning=f"Analyzed {len(threat_frames)} threat frames. Found {len(potential_false_positives)} potential false positives ({false_positive_rate:.3f} rate). Threshold: {self.quality_thresholds['max_false_positive_rate']}",
            output_data={
                'false_positive_rate': false_positive_rate,
                'false_positive_score': false_positive_score,
                'potential_false_positives': potential_false_positives
            },
            confidence=false_positive_score,
            duration_ms=duration_ms
        )
        
        # Update metrics
        self.qa_metrics['false_positives_identified'] += len(potential_false_positives)
        
        return {
            'false_positive_rate': false_positive_rate,
            'false_positive_score': false_positive_score,
            'potential_false_positives': potential_false_positives
        }
    
    async def _detect_false_negatives(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect potential false negatives in threat detection"""
        
        start_time = datetime.utcnow()
        
        non_threat_frames = [f for f in frame_analyses if f.get('threat_level', '').lower() not in ['high', 'medium']]
        potential_false_negatives = []
        
        for frame in non_threat_frames:
            frame_id = frame.get('frame_id')
            confidence = frame.get('confidence', 0.0)
            analysis_result = frame.get('analysis_result', {})
            
            # Check for indicators of false negatives
            false_negative_indicators = []
            
            # High confidence but no threat
            if confidence > 0.8:
                false_negative_indicators.append("High confidence but no threat detected")
            
            # Check analysis text for missed threats
            if 'gpt4o_enhanced' in analysis_result:
                gpt4o_result = analysis_result['gpt4o_enhanced']
                if 'enhanced_analysis' in gpt4o_result:
                    analysis_text = str(gpt4o_result['enhanced_analysis']).lower()
                    
                    # Look for threat-related keywords
                    threat_keywords = ['suspicious', 'unusual', 'concealing', 'hiding', 'grabbing', 'taking']
                    found_threat_keywords = [word for word in threat_keywords if word in analysis_text]
                    
                    if found_threat_keywords:
                        false_negative_indicators.append(f"Analysis contains threat keywords: {found_threat_keywords}")
            
            # People detected but no threat
            people_detected = analysis_result.get('people_detected', 0)
            if people_detected > 1:
                false_negative_indicators.append(f"Multiple people detected ({people_detected}) but no threat")
            
            # If multiple indicators found, flag as potential false negative
            if len(false_negative_indicators) >= 2:
                potential_false_negatives.append({
                    'frame_id': frame_id,
                    'confidence': confidence,
                    'people_detected': people_detected,
                    'indicators': false_negative_indicators
                })
        
        false_negative_rate = len(potential_false_negatives) / len(non_threat_frames) if non_threat_frames else 0.0
        false_negative_score = 1.0 - min(false_negative_rate, 1.0)
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="false_negative_detection",
            description="Detecting potential false negatives in threat detection",
            input_data={'non_threat_frames': len(non_threat_frames)},
            reasoning=f"Analyzed {len(non_threat_frames)} non-threat frames. Found {len(potential_false_negatives)} potential false negatives ({false_negative_rate:.3f} rate)",
            output_data={
                'false_negative_rate': false_negative_rate,
                'false_negative_score': false_negative_score,
                'potential_false_negatives': potential_false_negatives
            },
            confidence=false_negative_score,
            duration_ms=duration_ms
        )
        
        # Update metrics
        self.qa_metrics['false_negatives_identified'] += len(potential_false_negatives)
        
        return {
            'false_negative_rate': false_negative_rate,
            'false_negative_score': false_negative_score,
            'potential_false_negatives': potential_false_negatives
        }
    
    async def _check_consistency(self, frame_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check consistency across frame analyses"""
        
        start_time = datetime.utcnow()
        
        consistency_issues = []
        
        # Check for inconsistent threat levels in sequence
        threat_levels = [f.get('threat_level', 'low') for f in frame_analyses]
        
        # Look for isolated threat frames (threat surrounded by non-threats)
        for i, frame in enumerate(frame_analyses):
            if frame.get('threat_level', '').lower() in ['high', 'medium']:
                # Check neighbors
                prev_threat = i > 0 and frame_analyses[i-1].get('threat_level', '').lower() in ['high', 'medium']
                next_threat = i < len(frame_analyses) - 1 and frame_analyses[i+1].get('threat_level', '').lower() in ['high', 'medium']
                
                if not prev_threat and not next_threat:
                    consistency_issues.append({
                        'type': 'isolated_threat',
                        'frame_id': frame.get('frame_id'),
                        'description': "Threat detected in isolated frame with no surrounding threats"
                    })
        
        # Check confidence consistency
        confidence_scores = [f.get('confidence', 0.0) for f in frame_analyses]
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            std_confidence = (sum((x - avg_confidence) ** 2 for x in confidence_scores) / len(confidence_scores)) ** 0.5
            
            if std_confidence > 0.3:
                consistency_issues.append({
                    'type': 'confidence_variance',
                    'description': f"High variance in confidence scores (std: {std_confidence:.3f})"
                })
        
        consistency_score = 1.0 - min(len(consistency_issues) * 0.1, 1.0)
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="consistency_check",
            description="Checking consistency across frame analyses",
            input_data={'frame_count': len(frame_analyses)},
            reasoning=f"Checked consistency across {len(frame_analyses)} frames. Found {len(consistency_issues)} consistency issues",
            output_data={
                'consistency_score': consistency_score,
                'consistency_issues': consistency_issues
            },
            confidence=consistency_score,
            duration_ms=duration_ms
        )
        
        return {
            'consistency_score': consistency_score,
            'consistency_issues': consistency_issues
        }
    
    async def _generate_quality_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall quality assessment"""
        
        start_time = datetime.utcnow()
        
        # Calculate overall quality score
        completeness_score = results['completeness']['completeness_score']
        confidence_score = results['confidence']['confidence_score']
        false_positive_score = results['false_positives']['false_positive_score']
        false_negative_score = results['false_negatives']['false_negative_score']
        consistency_score = results['consistency']['consistency_score']
        
        overall_quality_score = (
            completeness_score * 0.25 +
            confidence_score * 0.25 +
            false_positive_score * 0.2 +
            false_negative_score * 0.15 +
            consistency_score * 0.15
        )
        
        # Collect all issues
        all_issues = []
        all_issues.extend(results['completeness']['issues'])
        
        if results['confidence']['low_confidence_frames']:
            all_issues.append(f"Low confidence in {len(results['confidence']['low_confidence_frames'])} frames")
        
        if results['false_positives']['potential_false_positives']:
            all_issues.append(f"Potential false positives: {len(results['false_positives']['potential_false_positives'])}")
        
        if results['false_negatives']['potential_false_negatives']:
            all_issues.append(f"Potential false negatives: {len(results['false_negatives']['potential_false_negatives'])}")
        
        if results['consistency']['consistency_issues']:
            all_issues.append(f"Consistency issues: {len(results['consistency']['consistency_issues'])}")
        
        # Generate recommendations
        recommendations = []
        
        if completeness_score < 0.8:
            recommendations.append("Improve data completeness and metadata collection")
        
        if confidence_score < 0.8:
            recommendations.append("Review and improve confidence calibration")
        
        if false_positive_score < 0.8:
            recommendations.append("Refine threat detection to reduce false positives")
        
        if false_negative_score < 0.8:
            recommendations.append("Enhance sensitivity to reduce false negatives")
        
        if consistency_score < 0.8:
            recommendations.append("Improve temporal consistency in analysis")
        
        # Determine if validation passes
        is_valid = overall_quality_score >= 0.7
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="quality_assessment",
            description="Generating overall quality assessment",
            input_data={'component_scores': results},
            reasoning=f"Calculated overall quality score {overall_quality_score:.3f} from 5 components. Found {len(all_issues)} issues. Generated {len(recommendations)} recommendations.",
            output_data={
                'overall_quality_score': overall_quality_score,
                'is_valid': is_valid,
                'issues_found': all_issues,
                'recommendations': recommendations,
                'component_scores': {
                    'completeness': completeness_score,
                    'confidence': confidence_score,
                    'false_positive': false_positive_score,
                    'false_negative': false_negative_score,
                    'consistency': consistency_score
                }
            },
            confidence=overall_quality_score,
            duration_ms=duration_ms
        )
        
        return {
            'overall_quality_score': overall_quality_score,
            'is_valid': is_valid,
            'confidence_score': overall_quality_score,
            'issues_found': all_issues,
            'recommendations': recommendations,
            'quality_score': overall_quality_score,
            'validation_details': {
                'component_scores': {
                    'completeness': completeness_score,
                    'confidence': confidence_score,
                    'false_positive': false_positive_score,
                    'false_negative': false_negative_score,
                    'consistency': consistency_score
                },
                'detailed_results': results
            }
        }
    
    def get_qa_metrics(self) -> Dict[str, Any]:
        """Get QA-specific metrics"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'qa_metrics': self.qa_metrics,
            'quality_thresholds': self.quality_thresholds,
            'general_metrics': self.get_metrics()
        }
