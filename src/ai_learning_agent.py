"""
ai_learning_agent.py - Advanced AI Learning Agent using LangGraph

This agent uses LangGraph for sophisticated reasoning, memory management,
and learning from video analysis. It implements a stateful agent that can:
- Learn patterns from video analysis
- Remember incidents and behaviors across videos
- Reason about threats and similarities
- Adapt threat detection over time
- Provide contextual analysis with memory
"""

import os

# Clear proxy settings process-wide if they cause httpx/openai client validation errors
for env_var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(env_var, None)

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Annotated
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib
import re
import pickle

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.schema import BaseRetriever

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """State for the AI learning agent"""
    current_video_id: str
    frame_analyses: List[Dict[str, Any]]
    video_metadata: Dict[str, Any]
    learned_patterns: List[Dict[str, Any]]
    incidents: List[Dict[str, Any]]
    reasoning_chain: List[str]
    confidence_scores: List[float]
    adaptation_level: float
    memory_context: Dict[str, Any]
    cross_video_connections: List[str]
    threat_assessment: Dict[str, Any]
    next_action: str

class SecurityPattern(BaseModel):
    """Security pattern learned from incidents"""
    pattern_id: str = Field(description="Unique pattern identifier")
    pattern_type: str = Field(description="Type of pattern: behavioral, visual, temporal, contextual")
    description: str = Field(description="Pattern description")
    confidence: float = Field(description="Pattern confidence score")
    frequency: int = Field(description="How many times this pattern was observed")
    first_seen: datetime = Field(description="When pattern was first observed")
    last_seen: datetime = Field(description="When pattern was last observed")
    associated_threats: List[str] = Field(description="Threat levels associated with this pattern")
    context_tags: List[str] = Field(description="Contextual tags for this pattern")
    success_rate: float = Field(description="Pattern success rate")
    false_positive_rate: float = Field(description="Pattern false positive rate")

class SecurityIncident(BaseModel):
    """Security incident record"""
    incident_id: str = Field(description="Unique incident identifier")
    video_id: str = Field(description="Video where incident occurred")
    incident_type: str = Field(description="Type of incident")
    severity: str = Field(description="Incident severity level")
    description: str = Field(description="Incident description")
    timestamp: datetime = Field(description="When incident occurred")
    location: str = Field(description="Location of incident")
    camera_id: str = Field(description="Camera that captured incident")
    frames_involved: List[str] = Field(description="Frames involved in incident")
    patterns_detected: List[str] = Field(description="Patterns detected in incident")
    outcome: str = Field(description="Incident outcome")
    response_time: float = Field(description="Time to detect incident in seconds")
    confidence: float = Field(description="Incident confidence score")
    metadata: Dict[str, Any] = Field(description="Additional incident metadata")

class VideoLearningProfile(BaseModel):
    """Learning profile for a specific video"""
    video_id: str = Field(description="Video identifier")
    total_frames: int = Field(description="Total frames in video")
    threat_frames: int = Field(description="Number of threat frames")
    patterns_detected: List[str] = Field(description="Patterns detected in video")
    unique_behaviors: Set[str] = Field(description="Unique behaviors observed")
    threat_timeline: List[Tuple[float, str]] = Field(description="Timeline of threats")
    confidence_trend: List[float] = Field(description="Confidence scores over time")
    learning_score: float = Field(description="Overall learning score")
    adaptation_level: int = Field(description="Adaptation level (0-100)")
    created_at: datetime = Field(description="When profile was created")
    updated_at: datetime = Field(description="When profile was last updated")

class AILearningAgent:
    """Advanced AI Learning Agent using LangGraph"""
    
    def __init__(self, knowledge_base_path: str = "data/ai_knowledge_base", openai_api_key: str = None):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM — learning agent uses NVIDIA NIM or Groq for reasoning.
        from src.config import settings
        
        nvidia_api_key = getattr(settings, 'NVIDIA_API_KEY', '') or os.environ.get("NVIDIA_API_KEY", "")
        nvidia_model = getattr(settings, 'NVIDIA_MODEL', '') or os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
        groq_api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        llm_provider = os.environ.get("AGENT_LLM_PROVIDER", getattr(settings, 'AGENT_LLM_PROVIDER', 'nvidia')).lower()
        
        if nvidia_api_key and llm_provider == "nvidia":
            self.llm = ChatOpenAI(
                model=nvidia_model,
                base_url="https://integrate.api.nvidia.com/v1",
                openai_api_key=nvidia_api_key,
                temperature=0.1
            )
        elif groq_api_key:
            self.llm = ChatOpenAI(
                model="llama-3.3-70b-versatile",
                base_url="https://api.groq.com/openai/v1",
                openai_api_key=groq_api_key,
                temperature=0.1
            )
        else:
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.1,
                openai_api_key=openai_api_key or "sk-proj-..."
            )
        
        # Knowledge storage
        self.patterns_file = self.knowledge_base_path / "patterns.json"
        self.incidents_file = self.knowledge_base_path / "incidents.json"
        self.profiles_file = self.knowledge_base_path / "video_profiles.json"
        self.memory_file = self.knowledge_base_path / "agent_memory.pkl"
        self.reasoning_file = self.knowledge_base_path / "reasoning_chains.json"
        
        # Learning data
        self.patterns: Dict[str, SecurityPattern] = {}
        self.incidents: Dict[str, SecurityIncident] = {}
        self.video_profiles: Dict[str, VideoLearningProfile] = {}
        self.agent_memory: Dict[str, Any] = {}
        self.reasoning_chains: Dict[str, List[str]] = {}
        
        # ML components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.pattern_vectors = None
        self.is_trained = False
        
        # Learning parameters
        self.min_pattern_frequency = 3
        self.confidence_threshold = 0.7
        self.adaptation_rate = 0.1
        
        # Build LangGraph
        self.graph = self._build_agent_graph()
        
        # Load existing knowledge
        self.load_knowledge()
        
        logger.info(f"AI Learning Agent initialized with {len(self.patterns)} patterns, {len(self.incidents)} incidents")
    
    def _build_agent_graph(self) -> StateGraph:
        """Build the LangGraph for the AI agent"""
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_video", self._analyze_video_node)
        workflow.add_node("extract_patterns", self._extract_patterns_node)
        workflow.add_node("reason_about_threats", self._reason_about_threats_node)
        workflow.add_node("update_memory", self._update_memory_node)
        workflow.add_node("adapt_detection", self._adapt_detection_node)
        workflow.add_node("generate_insights", self._generate_insights_node)
        
        # Add edges
        workflow.set_entry_point("analyze_video")
        workflow.add_edge("analyze_video", "extract_patterns")
        workflow.add_edge("extract_patterns", "reason_about_threats")
        workflow.add_edge("reason_about_threats", "update_memory")
        workflow.add_edge("update_memory", "adapt_detection")
        workflow.add_edge("adapt_detection", "generate_insights")
        workflow.add_edge("generate_insights", END)
        
        return workflow.compile()
    
    async def _analyze_video_node(self, state: AgentState) -> AgentState:
        """Analyze video for patterns and threats"""
        logger.info(f"Analyzing video {state.current_video_id}")
        
        # Extract basic statistics
        total_frames = len(state.frame_analyses)
        threat_frames = len([f for f in state.frame_analyses if f.get('threat_level', '').lower() in ['high', 'medium']])
        
        # Extract patterns from frame analyses
        patterns_detected = []
        unique_behaviors = set()
        threat_timeline = []
        confidence_scores = []
        
        for frame in state.frame_analyses:
            # Extract behaviors from analysis
            analysis_result = frame.get('analysis_result', {})
            
            if 'gpt4o_enhanced' in analysis_result:
                gpt4o_result = analysis_result['gpt4o_enhanced']
                if 'enhanced_analysis' in gpt4o_result:
                    enhanced_text = str(gpt4o_result['enhanced_analysis'])
                    
                    # Extract behaviors using regex patterns
                    behaviors = self._extract_behaviors(enhanced_text)
                    unique_behaviors.update(behaviors)
                    
                    # Extract patterns
                    frame_patterns = self._extract_frame_patterns(enhanced_text)
                    patterns_detected.extend(frame_patterns)
            
            # Build threat timeline
            if frame.get('threat_level', '').lower() in ['high', 'medium']:
                threat_timeline.append((frame.get('timestamp', 0), frame.get('threat_level', '')))
            
            # Track confidence
            confidence_scores.append(frame.get('confidence', 0.0))
        
        # Update state
        state.learned_patterns = patterns_detected
        state.confidence_scores = confidence_scores
        state.memory_context['unique_behaviors'] = list(unique_behaviors)
        state.memory_context['threat_timeline'] = threat_timeline
        state.memory_context['total_frames'] = total_frames
        state.memory_context['threat_frames'] = threat_frames
        
        # Add reasoning step
        reasoning = f"Analyzed {total_frames} frames from {state.current_video_id}. Found {threat_frames} threat frames and {len(patterns_detected)} patterns."
        state.reasoning_chain.append(reasoning)
        
        return state
    
    async def _extract_patterns_node(self, state: AgentState) -> AgentState:
        """Extract and categorize patterns using AI reasoning"""
        logger.info(f"Extracting patterns from {state.current_video_id}")
        
        # Use LLM to analyze patterns
        pattern_analysis_prompt = f"""
        Analyze the following patterns detected in video {state.current_video_id}:
        
        Patterns detected: {state.learned_patterns}
        Video context: {state.video_metadata}
        Unique behaviors: {state.memory_context.get('unique_behaviors', [])}
        
        Please:
        1. Categorize these patterns (behavioral, visual, temporal, contextual)
        2. Identify which patterns are most significant for security
        3. Suggest relationships between patterns
        4. Identify any novel patterns not seen before
        
        Respond with a structured analysis.
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a security pattern analysis expert."),
                HumanMessage(content=pattern_analysis_prompt)
            ])
            
            # Parse the response to extract structured pattern information
            pattern_insights = self._parse_pattern_response(response.content)
            state.memory_context['pattern_insights'] = pattern_insights
            
            # Add reasoning step
            reasoning = f"AI analysis identified {len(pattern_insights)} significant patterns with security implications."
            state.reasoning_chain.append(reasoning)
            
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            state.memory_context['pattern_insights'] = []
        
        return state
    
    async def _reason_about_threats_node(self, state: AgentState) -> AgentState:
        """Reason about threats using AI and historical context"""
        logger.info(f"Reasoning about threats in {state.current_video_id}")
        
        # Get similar videos from memory
        similar_videos = self._find_similar_videos(state.current_video_id)
        state.cross_video_connections = similar_videos
        
        # Use LLM for threat reasoning
        threat_reasoning_prompt = f"""
        Analyze the threat level for video {state.current_video_id} considering:
        
        Current video data:
        - Total frames: {state.memory_context.get('total_frames', 0)}
        - Threat frames: {state.memory_context.get('threat_frames', 0)}
        - Patterns: {state.learned_patterns}
        - Confidence trend: {state.confidence_scores}
        
        Historical context:
        - Similar videos: {similar_videos}
        - Known patterns: {list(self.patterns.keys())[:5]}
        
        Please provide:
        1. Overall threat assessment (low/medium/high/critical)
        2. Reasoning behind the assessment
        3. Confidence in the assessment
        4. Recommended actions
        5. Learning insights for future videos
        
        Respond with structured threat analysis.
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a security threat analysis expert with deep knowledge of behavioral patterns."),
                HumanMessage(content=threat_reasoning_prompt)
            ])
            
            # Parse threat assessment
            threat_assessment = self._parse_threat_response(response.content)
            state.threat_assessment = threat_assessment
            
            # Add reasoning step
            reasoning = f"AI reasoning determined threat level: {threat_assessment.get('threat_level', 'unknown')} with confidence {threat_assessment.get('confidence', 0):.2f}"
            state.reasoning_chain.append(reasoning)
            
        except Exception as e:
            logger.error(f"Threat reasoning failed: {e}")
            state.threat_assessment = {'threat_level': 'medium', 'confidence': 0.5, 'reasoning': 'Analysis failed'}
        
        return state
    
    async def _update_memory_node(self, state: AgentState) -> AgentState:
        """Update agent memory with new learnings"""
        logger.info(f"Updating memory with learnings from {state.current_video_id}")
        
        # Create incidents for threat frames
        threat_frames = [f for f in state.frame_analyses if f.get('threat_level', '').lower() in ['high', 'medium']]
        
        for frame in threat_frames:
            incident_id = f"incident_{state.current_video_id}_{frame.get('frame_number', 0)}"
            
            # Extract patterns from frame
            analysis_result = frame.get('analysis_result', {})
            patterns_detected = []
            
            if 'gpt4o_enhanced' in analysis_result:
                gpt4o_result = analysis_result['gpt4o_enhanced']
                if 'enhanced_analysis' in gpt4o_result:
                    enhanced_text = str(gpt4o_result['enhanced_analysis'])
                    patterns_detected = self._extract_frame_patterns(enhanced_text)
            
            # Create incident
            incident = SecurityIncident(
                incident_id=incident_id,
                video_id=state.current_video_id,
                incident_type="threat_detected",
                severity=frame.get('threat_level', 'medium'),
                description=f"AI-identified threat at frame {frame.get('frame_number', 0)}",
                timestamp=datetime.utcnow(),
                location=state.video_metadata.get('location', 'unknown'),
                camera_id=state.video_metadata.get('camera_id', 'unknown'),
                frames_involved=[frame.get('frame_id', '')],
                patterns_detected=patterns_detected,
                outcome="investigating",
                response_time=frame.get('timestamp', 0.0),
                confidence=frame.get('confidence', 0.0),
                metadata={
                    'frame_number': frame.get('frame_number', 0),
                    'people_detected': frame.get('people_detected', 0),
                    'objects_detected': frame.get('objects_detected', []),
                    'ai_reasoning': state.reasoning_chain[-1] if state.reasoning_chain else ''
                }
            )
            
            self.incidents[incident_id] = incident
            
            # Update pattern frequency
            for pattern_name in patterns_detected:
                self._update_pattern_frequency(pattern_name, incident)
        
        # Update video profile
        self._update_video_profile(state)
        
        # Store reasoning chain
        self.reasoning_chains[state.current_video_id] = state.reasoning_chain
        
        # Add reasoning step
        reasoning = f"Memory updated with {len(threat_frames)} incidents and {len(state.learned_patterns)} patterns."
        state.reasoning_chain.append(reasoning)
        
        return state
    
    async def _adapt_detection_node(self, state: AgentState) -> AgentState:
        """Adapt threat detection based on learnings"""
        logger.info(f"Adapting detection based on learnings from {state.current_video_id}")
        
        # Calculate adaptation level
        base_adaptation = state.memory_context.get('threat_frames', 0) / max(state.memory_context.get('total_frames', 1), 1)
        pattern_diversity = len(set(state.learned_patterns))
        confidence_avg = sum(state.confidence_scores) / len(state.confidence_scores) if state.confidence_scores else 0
        
        # Weighted adaptation calculation
        state.adaptation_level = min(
            (base_adaptation * 0.4 + pattern_diversity * 0.3 + confidence_avg * 0.3) * 100,
            100
        )
        
        # Retrain pattern recognition if enough data
        if len(self.patterns) >= 5:
            self._train_pattern_recognition()
        
        # Add reasoning step
        reasoning = f"Adaptation level set to {state.adaptation_level:.1f}% based on threat ratio, pattern diversity, and confidence."
        state.reasoning_chain.append(reasoning)
        
        return state
    
    async def _generate_insights_node(self, state: AgentState) -> AgentState:
        """Generate final insights and recommendations"""
        logger.info(f"Generating insights for {state.current_video_id}")
        
        # Use LLM to generate comprehensive insights
        insights_prompt = f"""
        Generate comprehensive security insights for video {state.current_video_id}:
        
        Analysis Summary:
        - Video: {state.current_video_id}
        - Location: {state.video_metadata.get('location', 'unknown')}
        - Total frames: {state.memory_context.get('total_frames', 0)}
        - Threat frames: {state.memory_context.get('threat_frames', 0)}
        - Patterns detected: {state.learned_patterns}
        - Adaptation level: {state.adaptation_level:.1f}%
        
        AI Reasoning Chain:
        {' -> '.join(state.reasoning_chain)}
        
        Please provide:
        1. Executive summary of security findings
        2. Key risk factors identified
        3. Behavioral patterns of concern
        4. Recommendations for security improvement
        5. Learning insights for future monitoring
        6. Confidence level in findings
        
        Format as a comprehensive security report.
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a senior security analyst providing comprehensive threat assessments."),
                HumanMessage(content=insights_prompt)
            ])
            
            # Store insights
            state.memory_context['comprehensive_insights'] = response.content
            
            # Add final reasoning step
            reasoning = f"Generated comprehensive security insights with {len(response.content)} characters of analysis."
            state.reasoning_chain.append(reasoning)
            
        except Exception as e:
            logger.error(f"Insights generation failed: {e}")
            state.memory_context['comprehensive_insights'] = "Insights generation failed"
        
        return state
    
    async def analyze_video_with_ai(self, video_id: str, frame_analyses: List[Dict[str, Any]], 
                                   video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze video using the AI agent"""
        logger.info(f"Starting AI analysis for video {video_id}")
        
        # Initialize state
        initial_state = AgentState(
            current_video_id=video_id,
            frame_analyses=frame_analyses,
            video_metadata=video_metadata,
            learned_patterns=[],
            incidents=[],
            reasoning_chain=[],
            confidence_scores=[],
            adaptation_level=0.0,
            memory_context={},
            cross_video_connections=[],
            threat_assessment={},
            next_action=""
        )
        
        # Run the agent graph
        try:
            result = await self.graph.ainvoke(initial_state)
            
            # Convert AgentState to dict if needed
            if hasattr(result, 'dict'):
                result_dict = result.dict()
            else:
                result_dict = result
            
            # Save knowledge
            self.save_knowledge()
            
            return {
                'video_id': video_id,
                'learning_score': result_dict.get('adaptation_level', 0) / 100,
                'adaptation_level': result_dict.get('adaptation_level', 0),
                'patterns_detected': result_dict.get('learned_patterns', []),
                'threat_assessment': result_dict.get('threat_assessment', {}),
                'reasoning_chain': result_dict.get('reasoning_chain', []),
                'cross_video_connections': result_dict.get('cross_video_connections', []),
                'comprehensive_insights': result_dict.get('memory_context', {}).get('comprehensive_insights', ''),
                'memory_context': result_dict.get('memory_context', {}),
                'incident_count': len([f for f in frame_analyses if f.get('threat_level', '').lower() in ['high', 'medium']]),
                'total_frames': len(frame_analyses)
            }
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {
                'video_id': video_id,
                'error': str(e),
                'learning_score': 0.0,
                'adaptation_level': 0
            }
    
    def _extract_behaviors(self, text: str) -> List[str]:
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
    
    def _extract_frame_patterns(self, text: str) -> List[str]:
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
    
    def _parse_pattern_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM pattern response into structured data"""
        # Simplified parsing - in production, use more sophisticated parsing
        insights = []
        
        # Extract numbered items or bullet points
        lines = response.split('\n')
        for line in lines:
            if line.strip() and (line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '-', '*'))):
                insights.append({
                    'insight': line.strip(),
                    'category': 'pattern_analysis',
                    'confidence': 0.8
                })
        
        return insights
    
    def _parse_threat_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM threat response into structured data"""
        # Simplified parsing - in production, use more sophisticated parsing
        threat_assessment = {
            'threat_level': 'medium',
            'confidence': 0.7,
            'reasoning': response[:200] + '...' if len(response) > 200 else response,
            'recommendations': [],
            'full_response': response
        }
        
        # Look for threat level in response
        if 'high' in response.lower():
            threat_assessment['threat_level'] = 'high'
        elif 'critical' in response.lower():
            threat_assessment['threat_level'] = 'critical'
        elif 'low' in response.lower():
            threat_assessment['threat_level'] = 'low'
        
        return threat_assessment
    
    def _find_similar_videos(self, video_id: str) -> List[str]:
        """Find videos similar to the given video"""
        similar_videos = []
        
        if video_id in self.video_profiles:
            current_profile = self.video_profiles[video_id]
            
            for other_video_id, other_profile in self.video_profiles.items():
                if other_video_id != video_id:
                    # Calculate pattern similarity
                    patterns1 = set(current_profile.patterns_detected)
                    patterns2 = set(other_profile.patterns_detected)
                    
                    if patterns1 and patterns2:
                        intersection = len(patterns1.intersection(patterns2))
                        union = len(patterns1.union(patterns2))
                        similarity = intersection / union if union > 0 else 0
                        
                        if similarity > 0.3:  # Similarity threshold
                            similar_videos.append(other_video_id)
        
        return similar_videos
    
    def _update_pattern_frequency(self, pattern_name: str, incident: SecurityIncident):
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
    
    def _update_video_profile(self, state: AgentState):
        """Update video learning profile"""
        video_id = state.current_video_id
        
        if video_id in self.video_profiles:
            profile = self.video_profiles[video_id]
            profile.total_frames = len(state.frame_analyses)
            profile.threat_frames = state.memory_context.get('threat_frames', 0)
            profile.patterns_detected = list(set(state.learned_patterns + profile.patterns_detected))
            profile.unique_behaviors.update(state.memory_context.get('unique_behaviors', []))
            profile.threat_timeline = state.memory_context.get('threat_timeline', [])
            profile.confidence_trend = state.confidence_scores
            profile.updated_at = datetime.utcnow()
        else:
            profile = VideoLearningProfile(
                video_id=video_id,
                total_frames=len(state.frame_analyses),
                threat_frames=state.memory_context.get('threat_frames', 0),
                patterns_detected=state.learned_patterns,
                unique_behaviors=set(state.memory_context.get('unique_behaviors', [])),
                threat_timeline=state.memory_context.get('threat_timeline', []),
                confidence_trend=state.confidence_scores,
                learning_score=0.0,
                adaptation_level=int(state.adaptation_level),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        
        # Calculate learning score
        profile.learning_score = state.adaptation_level / 100
        
        self.video_profiles[video_id] = profile
    
    def _train_pattern_recognition(self):
        """Train pattern recognition model"""
        try:
            if len(self.patterns) < 5:
                return
            
            # Prepare training data
            pattern_texts = []
            for pattern in self.patterns.values():
                pattern_texts.append(pattern.description)
            
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
        similar_videos = self._find_similar_videos(video_id)
        
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
            },
            'ai_reasoning': self.reasoning_chains.get(video_id, [])[-1] if video_id in self.reasoning_chains else ''
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
            'reasoning_chains_count': len(self.reasoning_chains),
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
        else:
            insights['learning_trends'] = {
                'avg_learning_score': 0.0,
                'avg_adaptation_level': 0.0,
                'total_behaviors_learned': 0
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
    
    def load_knowledge(self):
        """Load existing knowledge base"""
        try:
            # Load patterns
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    for pattern_id, pattern_data in patterns_data.items():
                        # Convert datetime strings back to datetime objects
                        if 'first_seen' in pattern_data:
                            pattern_data['first_seen'] = datetime.fromisoformat(pattern_data['first_seen'])
                        if 'last_seen' in pattern_data:
                            pattern_data['last_seen'] = datetime.fromisoformat(pattern_data['last_seen'])
                        self.patterns[pattern_id] = SecurityPattern(**pattern_data)
            
            # Load incidents
            if self.incidents_file.exists():
                with open(self.incidents_file, 'r') as f:
                    incidents_data = json.load(f)
                    for incident_id, incident_data in incidents_data.items():
                        if 'timestamp' in incident_data:
                            incident_data['timestamp'] = datetime.fromisoformat(incident_data['timestamp'])
                        self.incidents[incident_id] = SecurityIncident(**incident_data)
            
            # Load video profiles
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                    for video_id, profile_data in profiles_data.items():
                        if 'unique_behaviors' in profile_data:
                            profile_data['unique_behaviors'] = set(profile_data['unique_behaviors'])
                        if 'created_at' in profile_data:
                            profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                        if 'updated_at' in profile_data:
                            profile_data['updated_at'] = datetime.fromisoformat(profile_data['updated_at'])
                        if 'threat_timeline' in profile_data:
                            profile_data['threat_timeline'] = [tuple(t) for t in profile_data['threat_timeline']]
                        self.video_profiles[video_id] = VideoLearningProfile(**profile_data)
            
            # Load memory
            if self.memory_file.exists():
                with open(self.memory_file, 'rb') as f:
                    self.agent_memory = pickle.load(f)
            
            # Load reasoning chains
            if self.reasoning_file.exists():
                with open(self.reasoning_file, 'r') as f:
                    self.reasoning_chains = json.load(f)
            
            # Train pattern recognition if we have enough data
            if len(self.patterns) >= 5:
                self._train_pattern_recognition()
                
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
    
    def save_knowledge(self):
        """Save knowledge base to disk"""
        try:
            # Save patterns
            patterns_data = {}
            for pattern_id, pattern in self.patterns.items():
                pattern_dict = pattern.dict()
                pattern_dict['first_seen'] = pattern.first_seen.isoformat()
                pattern_dict['last_seen'] = pattern.last_seen.isoformat()
                patterns_data[pattern_id] = pattern_dict
            
            with open(self.patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save incidents
            incidents_data = {}
            for incident_id, incident in self.incidents.items():
                incident_dict = incident.dict()
                incident_dict['timestamp'] = incident.timestamp.isoformat()
                incidents_data[incident_id] = incident_dict
            
            with open(self.incidents_file, 'w') as f:
                json.dump(incidents_data, f, indent=2)
            
            # Save video profiles
            profiles_data = {}
            for video_id, profile in self.video_profiles.items():
                profile_dict = profile.dict()
                profile_dict['unique_behaviors'] = list(profile.unique_behaviors)
                profile_dict['created_at'] = profile.created_at.isoformat()
                profile_dict['updated_at'] = profile.updated_at.isoformat()
                profile_dict['threat_timeline'] = [list(t) for t in profile.threat_timeline]
                profiles_data[video_id] = profile_dict
            
            with open(self.profiles_file, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            # Save memory
            with open(self.memory_file, 'wb') as f:
                pickle.dump(self.agent_memory, f)
            
            # Save reasoning chains
            with open(self.reasoning_file, 'w') as f:
                json.dump(self.reasoning_chains, f, indent=2)
            
            logger.info("AI knowledge base saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

# Global AI learning agent instance
ai_learning_agent = AILearningAgent()
