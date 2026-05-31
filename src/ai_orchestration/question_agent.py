"""
question_agent.py - Question Agent with Reasoning

The Question Agent is responsible for:
- Answering natural language questions about security analysis
- Providing insights and explanations
- Generating reports and summaries
- Explaining AI reasoning and decisions
- Handling complex queries across multiple data sources
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
class QuestionResult:
    """Result of question processing"""
    answer: str
    confidence: float
    sources: List[str]
    reasoning: str
    related_questions: List[str]
    follow_up_actions: List[str]

class QuestionAgent(BaseAgent):
    """Question Agent with comprehensive reasoning capabilities"""
    
    def __init__(self, openai_api_key: str = None):
        super().__init__("question_agent", openai_api_key)
        
        # Question-specific metrics
        self.question_metrics = {
            'questions_answered': 0,
            'high_confidence_answers': 0,
            'complex_queries_handled': 0,
            'reports_generated': 0,
            'insights_provided': 0,
            'avg_response_time_ms': 0.0,
            'question_types': {}
        }
        
        # Question categories
        self.question_categories = {
            'threat_analysis': [
                'threat', 'risk', 'danger', 'security', 'suspicious',
                'attack', 'breach', 'incident', 'alert'
            ],
            'behavioral_analysis': [
                'behavior', 'action', 'movement', 'pattern', 'activity',
                'person', 'individual', 'group', 'coordination'
            ],
            'temporal_analysis': [
                'timeline', 'sequence', 'progression', 'when', 'time',
                'duration', 'before', 'after', 'during'
            ],
            'data_quality': [
                'quality', 'accuracy', 'confidence', 'reliability',
                'false positive', 'false negative', 'validation'
            ],
            'general_inquiry': [
                'what', 'how', 'why', 'explain', 'describe',
                'summary', 'overview', 'report'
            ]
        }
        
        # Knowledge base cache
        self.knowledge_cache = {}
        self.cache_expiry = 3600  # 1 hour
    
    def get_system_prompt(self) -> str:
        """Get system prompt for question agent"""
        return """
        You are a Security Question Answering Agent specializing in drone security analysis.
        
        Your responsibilities:
        1. Answer natural language questions about security analysis results
        2. Provide clear explanations of AI reasoning and decisions
        3. Generate comprehensive reports and summaries
        4. Explain complex security concepts in simple terms
        5. Suggest follow-up actions and related questions
        6. Provide insights based on analysis data
        
        You should be:
        - Clear and concise in your answers
        - Provide specific evidence and reasoning
        - Explain complex concepts simply
        - Suggest practical actions when appropriate
        - Always cite your sources and confidence levels
        """
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process question and provide comprehensive answer"""
        start_time = datetime.utcnow()
        
        try:
            self.update_status(AgentStatus.PROCESSING, "Processing security question")
            
            # Extract data
            question = input_data.get('question', '')
            context = input_data.get('context', {})
            analysis_data = input_data.get('analysis_data', {})
            qa_data = input_data.get('qa_data', {})
            
            # Step 1: Question classification
            question_classification = await self._classify_question(question)
            
            # Step 2: Intent understanding
            intent_analysis = await self._understand_intent(question, context)
            
            # Step 3: Knowledge retrieval
            knowledge_retrieval = await self._retrieve_knowledge(question, question_classification, analysis_data, qa_data)
            
            # Step 4: Answer generation
            answer_generation = await self._generate_answer(question, question_classification, intent_analysis, knowledge_retrieval)
            
            # Step 5: Source verification
            source_verification = await self._verify_sources(answer_generation, knowledge_retrieval)
            
            # Step 6: Follow-up suggestions
            follow_up_suggestions = await self._generate_follow_up_suggestions(question, answer_generation, context)
            
            # Calculate duration
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Update metrics
            self.question_metrics['questions_answered'] += 1
            self.question_metrics['avg_response_time_ms'] = (
                (self.question_metrics['avg_response_time_ms'] * (self.question_metrics['questions_answered'] - 1) + duration_ms) /
                self.question_metrics['questions_answered']
            )
            
            if answer_generation['confidence'] > 0.8:
                self.question_metrics['high_confidence_answers'] += 1
            
            if question_classification['complexity'] == 'high':
                self.question_metrics['complex_queries_handled'] += 1
            
            # Track question types
            q_type = question_classification['category']
            self.question_metrics['question_types'][q_type] = self.question_metrics['question_types'].get(q_type, 0) + 1
            
            # Add final reasoning step
            self.add_reasoning_step(
                step_type="question_answer",
                description=f"Answering question: {question[:100]}...",
                input_data=input_data,
                reasoning=f"Classified as {question_classification['category']} question with {question_classification['complexity']} complexity. Generated answer with {answer_generation['confidence']:.3f} confidence.",
                output_data={
                    'answer': answer_generation['answer'],
                    'confidence': answer_generation['confidence'],
                    'sources': source_verification['verified_sources'],
                    'reasoning': answer_generation['reasoning'],
                    'follow_up_actions': follow_up_suggestions['actions'],
                    'related_questions': follow_up_suggestions['related_questions']
                },
                confidence=answer_generation['confidence'],
                duration_ms=duration_ms,
                metadata={
                    'question_category': question_classification['category'],
                    'question_complexity': question_classification['complexity'],
                    'sources_count': len(source_verification['verified_sources'])
                }
            )
            
            self.update_status(AgentStatus.COMPLETED)
            self.metrics['tasks_completed'] += 1
            
            return {
                'agent_id': self.agent_id,
                'agent_name': self.agent_name,
                'question_result': {
                    'question': question,
                    'answer': answer_generation['answer'],
                    'confidence': answer_generation['confidence'],
                    'sources': source_verification['verified_sources'],
                    'reasoning': answer_generation['reasoning'],
                    'follow_up_actions': follow_up_suggestions['actions'],
                    'related_questions': follow_up_suggestions['related_questions'],
                    'question_classification': question_classification,
                    'intent_analysis': intent_analysis
                },
                'reasoning_chain': self.get_reasoning_chain(),
                'metrics': self.get_metrics(),
                'question_metrics': self.question_metrics,
                'processing_time_ms': duration_ms
            }
            
        except Exception as e:
            self.update_status(AgentStatus.ERROR)
            self.metrics['tasks_failed'] += 1
            logger.error(f"Question processing failed: {e}")
            raise
    
    async def _classify_question(self, question: str) -> Dict[str, Any]:
        """Classify question type and complexity"""
        
        start_time = datetime.utcnow()
        
        question_lower = question.lower()
        
        # Determine category
        category = 'general_inquiry'
        for cat_name, keywords in self.question_categories.items():
            if any(keyword in question_lower for keyword in keywords):
                category = cat_name
                break
        
        # Determine complexity
        complexity_indicators = [
            'compare', 'analyze', 'evaluate', 'synthesize', 'comprehensive',
            'detailed', 'thorough', 'multiple', 'various', 'different'
        ]
        
        complexity = 'low'
        if any(indicator in question_lower for indicator in complexity_indicators):
            complexity = 'medium'
        
        # Check for multiple questions or complex requests
        if question.count('?') > 1 or 'and' in question_lower or 'or' in question_lower:
            complexity = 'high'
        
        # Determine question type
        question_type = 'factual'
        if any(word in question_lower for word in ['why', 'how', 'explain']):
            question_type = 'explanatory'
        elif any(word in question_lower for word in ['what if', 'would', 'could', 'should']):
            question_type = 'hypothetical'
        elif any(word in question_lower for word in ['compare', 'difference', 'better', 'worse']):
            question_type = 'comparative'
        elif any(word in question_lower for word in ['summary', 'overview', 'report']):
            question_type = 'summary'
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="question_classification",
            description="Classifying question type and complexity",
            input_data={'question': question},
            reasoning=f"Classified question as {category} with {complexity} complexity and {question_type} type",
            output_data={
                'category': category,
                'complexity': complexity,
                'type': question_type
            },
            confidence=0.9,
            duration_ms=duration_ms
        )
        
        return {
            'category': category,
            'complexity': complexity,
            'type': question_type,
            'keywords_found': [kw for kw in question_lower.split() if len(kw) > 3]
        }
    
    async def _understand_intent(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Understand user intent from question and context"""
        
        start_time = datetime.utcnow()
        
        # Extract key entities
        entities = {
            'time_references': [],
            'security_concepts': [],
            'data_sources': [],
            'actions_requested': []
        }
        
        question_lower = question.lower()
        
        # Time references
        time_words = ['today', 'yesterday', 'recent', 'latest', 'current', 'past', 'future']
        entities['time_references'] = [word for word in time_words if word in question_lower]
        
        # Security concepts
        security_concepts = ['threat', 'risk', 'incident', 'alert', 'pattern', 'behavior', 'anomaly']
        entities['security_concepts'] = [concept for concept in security_concepts if concept in question_lower]
        
        # Data sources
        data_sources = ['video', 'frame', 'analysis', 'report', 'camera', 'location']
        entities['data_sources'] = [source for source in data_sources if source in question_lower]
        
        # Actions requested
        action_words = ['explain', 'show', 'analyze', 'compare', 'summarize', 'identify', 'detect']
        entities['actions_requested'] = [action for action in action_words if action in question_lower]
        
        # Determine primary intent
        primary_intent = 'information_seeking'
        if any(action in entities['actions_requested'] for action in ['analyze', 'compare', 'identify']):
            primary_intent = 'analysis_request'
        elif any(action in entities['actions_requested'] for action in ['summarize', 'show']):
            primary_intent = 'summary_request'
        elif 'why' in question_lower or 'explain' in question_lower:
            primary_intent = 'explanation_request'
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="intent_understanding",
            description="Understanding user intent from question",
            input_data={'question': question, 'context_keys': list(context.keys())},
            reasoning=f"Identified primary intent as {primary_intent} with {len(entities['security_concepts'])} security concepts and {len(entities['actions_requested'])} actions",
            output_data={
                'primary_intent': primary_intent,
                'entities': entities
            },
            confidence=0.85,
            duration_ms=duration_ms
        )
        
        return {
            'primary_intent': primary_intent,
            'entities': entities,
            'context_relevance': len(context.keys()) > 0
        }
    
    async def _retrieve_knowledge(self, question: str, classification: Dict, analysis_data: Dict, qa_data: Dict) -> Dict[str, Any]:
        """Retrieve relevant knowledge from available data sources"""
        
        start_time = datetime.utcnow()
        
        knowledge_sources = []
        relevant_data = {}
        
        # Check analysis data
        if analysis_data:
            relevant_data['analysis_data'] = {
                'available': True,
                'components': list(analysis_data.keys()),
                'summary': self._summarize_analysis_data(analysis_data)
            }
            knowledge_sources.append('analysis_results')
        
        # Check QA data
        if qa_data:
            relevant_data['qa_data'] = {
                'available': True,
                'validation_results': qa_data.get('validation_results', {}),
                'quality_metrics': qa_data.get('quality_metrics', {})
            }
            knowledge_sources.append('quality_assurance')
        
        # Check cached knowledge
        cache_key = f"{classification['category']}_{hash(question)}"
        if cache_key in self.knowledge_cache:
            cached_data = self.knowledge_cache[cache_key]
            if (datetime.utcnow() - cached_data['timestamp']).seconds < self.cache_expiry:
                relevant_data['cached_knowledge'] = cached_data['data']
                knowledge_sources.append('cached_knowledge')
        
        # Generate knowledge summary
        knowledge_summary = await self._generate_knowledge_summary(question, relevant_data)
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="knowledge_retrieval",
            description="Retrieving relevant knowledge from data sources",
            input_data={'question_category': classification['category']},
            reasoning=f"Retrieved knowledge from {len(knowledge_sources)} sources: {knowledge_sources}",
            output_data={
                'knowledge_sources': knowledge_sources,
                'relevant_data': relevant_data,
                'knowledge_summary': knowledge_summary
            },
            confidence=0.8,
            duration_ms=duration_ms
        )
        
        return {
            'knowledge_sources': knowledge_sources,
            'relevant_data': relevant_data,
            'knowledge_summary': knowledge_summary
        }
    
    async def _generate_answer(self, question: str, classification: Dict, intent: Dict, knowledge: Dict) -> Dict[str, Any]:
        """Generate comprehensive answer"""
        
        start_time = datetime.utcnow()
        
        # Build context for LLM
        context_prompt = f"""
        Question: {question}
        
        Question Classification:
        - Category: {classification['category']}
        - Complexity: {classification['complexity']}
        - Type: {classification['type']}
        
        User Intent:
        - Primary Intent: {intent['primary_intent']}
        - Entities: {intent['entities']}
        
        Available Knowledge:
        {knowledge['knowledge_summary']}
        
        Please provide a comprehensive answer that:
        1. Directly addresses the question
        2. Provides specific evidence and reasoning
        3. Explains complex concepts clearly
        4. Includes confidence assessment
        5. Suggests practical implications
        """
        
        try:
            # Generate answer using LLM
            answer_text = await self.call_llm([context_prompt])
            
            # Extract reasoning from answer
            reasoning = self._extract_reasoning_from_answer(answer_text)
            
            # Calculate confidence based on available knowledge and answer quality
            confidence = self._calculate_answer_confidence(knowledge, answer_text)
            
            # Add reasoning step
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            self.add_reasoning_step(
                step_type="answer_generation",
                description="Generating comprehensive answer",
                input_data={'question_length': len(question), 'knowledge_sources': len(knowledge['knowledge_sources'])},
                reasoning=f"Generated answer with {confidence:.3f} confidence using {len(knowledge['knowledge_sources'])} knowledge sources",
                output_data={
                    'answer': answer_text,
                    'reasoning': reasoning,
                    'confidence': confidence
                },
                confidence=confidence,
                duration_ms=duration_ms
            )
            
            return {
                'answer': answer_text,
                'reasoning': reasoning,
                'confidence': confidence,
                'answer_length': len(answer_text)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                'answer': f"I apologize, but I encountered an error while processing your question. Please try rephrasing or contact support.",
                'reasoning': "Error occurred during answer generation",
                'confidence': 0.0,
                'answer_length': 0
            }
    
    async def _verify_sources(self, answer_generation: Dict, knowledge: Dict) -> Dict[str, Any]:
        """Verify and validate sources used in answer"""
        
        start_time = datetime.utcnow()
        
        verified_sources = []
        source_confidence = {}
        
        for source in knowledge['knowledge_sources']:
            source_validity = self._validate_source(source, knowledge['relevant_data'])
            verified_sources.append({
                'source': source,
                'validity': source_validity['validity'],
                'confidence': source_validity['confidence'],
                'details': source_validity['details']
            })
            source_confidence[source] = source_validity['confidence']
        
        # Calculate overall source confidence
        overall_source_confidence = sum(source_confidence.values()) / len(source_confidence) if source_confidence else 0.0
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="source_verification",
            description="Verifying sources used in answer",
            input_data={'sources_count': len(knowledge['knowledge_sources'])},
            reasoning=f"Verified {len(verified_sources)} sources with overall confidence {overall_source_confidence:.3f}",
            output_data={
                'verified_sources': verified_sources,
                'overall_source_confidence': overall_source_confidence
            },
            confidence=overall_source_confidence,
            duration_ms=duration_ms
        )
        
        return {
            'verified_sources': verified_sources,
            'overall_source_confidence': overall_source_confidence
        }
    
    async def _generate_follow_up_suggestions(self, question: str, answer: Dict, context: Dict) -> Dict[str, Any]:
        """Generate follow-up actions and related questions"""
        
        start_time = datetime.utcnow()
        
        # Generate follow-up actions
        actions_prompt = f"""
        Based on this question and answer, suggest 3-5 practical follow-up actions:
        
        Question: {question}
        Answer: {answer['answer'][:500]}...
        
        Suggest specific, actionable next steps for security personnel.
        """
        
        try:
            actions_response = await self.call_llm([actions_prompt])
            actions = self._parse_list_response(actions_response)
        except Exception as e:
            logger.error(f"Failed to generate follow-up actions: {e}")
            actions = ["Review detailed analysis", "Consult security team", "Monitor for similar patterns"]
        
        # Generate related questions
        questions_prompt = f"""
        Based on this question and answer, suggest 3-5 related questions that might be helpful:
        
        Original Question: {question}
        Answer: {answer['answer'][:500]}...
        
        Generate relevant follow-up questions.
        """
        
        try:
            questions_response = await self.call_llm([questions_prompt])
            related_questions = self._parse_list_response(questions_response)
        except Exception as e:
            logger.error(f"Failed to generate related questions: {e}")
            related_questions = ["What are the risk factors?", "How can we improve detection?", "What patterns should we monitor?"]
        
        # Add reasoning step
        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        self.add_reasoning_step(
            step_type="follow_up_generation",
            description="Generating follow-up suggestions",
            input_data={'question_length': len(question)},
            reasoning=f"Generated {len(actions)} follow-up actions and {len(related_questions)} related questions",
            output_data={
                'actions': actions,
                'related_questions': related_questions
            },
            confidence=0.8,
            duration_ms=duration_ms
        )
        
        return {
            'actions': actions,
            'related_questions': related_questions
        }
    
    def _summarize_analysis_data(self, analysis_data: Dict) -> str:
        """Summarize analysis data for context"""
        
        summary_parts = []
        
        if 'frame_results' in analysis_data:
            frame_results = analysis_data['frame_results']
            threat_frames = len([r for r in frame_results if r.get('threat_level') in ['high', 'medium']])
            summary_parts.append(f"Analyzed {len(frame_results)} frames with {threat_frames} threat detections")
        
        if 'risk_assessment' in analysis_data:
            risk = analysis_data['risk_assessment']
            summary_parts.append(f"Overall risk level: {risk.get('risk_level', 'unknown')} (score: {risk.get('overall_risk_score', 0):.3f})")
        
        if 'patterns' in analysis_data:
            patterns = analysis_data['patterns']
            summary_parts.append(f"Identified {len(patterns)} security patterns")
        
        return "; ".join(summary_parts) if summary_parts else "Analysis data available"
    
    async def _generate_knowledge_summary(self, question: str, relevant_data: Dict) -> str:
        """Generate summary of available knowledge"""
        
        summary_parts = []
        
        if 'analysis_data' in relevant_data:
            summary_parts.append("Analysis results available with frame-by-frame threat detection")
        
        if 'qa_data' in relevant_data:
            summary_parts.append("Quality assurance data available with validation metrics")
        
        if 'cached_knowledge' in relevant_data:
            summary_parts.append("Relevant cached knowledge available")
        
        return "; ".join(summary_parts) if summary_parts else "Limited knowledge available"
    
    def _extract_reasoning_from_answer(self, answer: str) -> str:
        """Extract reasoning from generated answer"""
        
        # Look for reasoning indicators
        reasoning_indicators = [
            "because", "since", "due to", "based on", "according to",
            "evidence suggests", "analysis shows", "data indicates"
        ]
        
        sentences = answer.split('.')
        reasoning_sentences = []
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in reasoning_indicators):
                reasoning_sentences.append(sentence.strip())
        
        return ". ".join(reasoning_sentences) if reasoning_sentences else "Answer based on available analysis data and security expertise"
    
    def _calculate_answer_confidence(self, knowledge: Dict, answer: str) -> float:
        """Calculate confidence in generated answer"""
        
        base_confidence = 0.5
        
        # Adjust for available knowledge sources
        knowledge_bonus = min(len(knowledge['knowledge_sources']) * 0.1, 0.3)
        
        # Adjust for answer length and quality
        length_bonus = min(len(answer) / 1000, 0.2)
        
        # Check for confidence indicators in answer
        confidence_indicators = ['confident', 'certain', 'clear', 'definitely']
        uncertainty_indicators = ['uncertain', 'maybe', 'possibly', 'might', 'could']
        
        answer_lower = answer.lower()
        confidence_adjustment = 0.0
        
        for indicator in confidence_indicators:
            if indicator in answer_lower:
                confidence_adjustment += 0.05
        
        for indicator in uncertainty_indicators:
            if indicator in answer_lower:
                confidence_adjustment -= 0.05
        
        # Calculate final confidence
        final_confidence = base_confidence + knowledge_bonus + length_bonus + confidence_adjustment
        
        return max(0.0, min(1.0, final_confidence))
    
    def _validate_source(self, source: str, relevant_data: Dict) -> Dict[str, Any]:
        """Validate a specific knowledge source"""
        
        if source == 'analysis_results':
            return {
                'validity': 'high',
                'confidence': 0.9,
                'details': 'Direct analysis results from video processing'
            }
        elif source == 'quality_assurance':
            return {
                'validity': 'high',
                'confidence': 0.85,
                'details': 'QA validation results with quality metrics'
            }
        elif source == 'cached_knowledge':
            return {
                'validity': 'medium',
                'confidence': 0.7,
                'details': 'Cached knowledge from previous queries'
            }
        else:
            return {
                'validity': 'low',
                'confidence': 0.5,
                'details': 'Unknown or unverified source'
            }
    
    def _parse_list_response(self, response: str) -> List[str]:
        """Parse numbered or bulleted list from LLM response"""
        
        items = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*', '•'))):
                # Remove list marker
                clean_line = re.sub(r'^[0-9.\-*\•]+\s*', '', line)
                if clean_line:
                    items.append(clean_line)
        
        return items if items else [response.strip()]
    
    def get_question_metrics(self) -> Dict[str, Any]:
        """Get question-specific metrics"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'question_metrics': self.question_metrics,
            'question_categories': self.question_categories,
            'general_metrics': self.get_metrics()
        }
