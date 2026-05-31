"""
orchestrator.py - AI Orchestration System

The orchestrator coordinates all AI agents and manages the overall workflow:
- Coordinates agent execution
- Manages data flow between agents
- Handles error recovery and retries
- Provides unified interface for agent operations
- Manages agent state and communication
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
import logging

from .base_agent import BaseAgent, AgentStatus
from .qa_agent import QAAgent
from .analysis_agent import AnalysisAgent
from .question_agent import QuestionAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    """Workflow types for orchestration"""
    ANALYSIS_ONLY = "analysis_only"
    ANALYSIS_WITH_QA = "analysis_with_qa"
    QUESTION_ANSWERING = "question_answering"
    COMPREHENSIVE = "comprehensive"

@dataclass
class WorkflowStep:
    """Single step in a workflow"""
    step_id: str
    agent_name: str
    step_type: str
    input_data: Dict[str, Any]
    dependencies: List[str]
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 60

@dataclass
class WorkflowResult:
    """Result of a workflow execution"""
    workflow_id: str
    workflow_type: WorkflowType
    status: str
    results: Dict[str, Any]
    agent_results: Dict[str, Any]
    reasoning_chains: Dict[str, List[Dict]]
    execution_time_ms: int
    errors: List[str]

class AIOrchestrator:
    """Main AI Orchestrator for coordinating all agents"""
    
    def __init__(self, openai_api_key: str = None):
        self.orchestrator_id = f"orchestrator_{uuid.uuid4().hex[:8]}"
        
        # Initialize agents
        self.agents = {
            'qa_agent': QAAgent(openai_api_key),
            'analysis_agent': AnalysisAgent(openai_api_key),
            'question_agent': QuestionAgent(openai_api_key)
        }
        
        # Workflow management
        self.active_workflows = {}
        self.workflow_history = []
        
        # Orchestration metrics
        self.orchestration_metrics = {
            'workflows_executed': 0,
            'workflows_completed': 0,
            'workflows_failed': 0,
            'total_execution_time_ms': 0,
            'agent_usage': {agent_name: 0 for agent_name in self.agents.keys()},
            'workflow_types': {}
        }
        
        logger.info(f"AI Orchestrator initialized with ID: {self.orchestrator_id}")
        logger.info(f"Available agents: {list(self.agents.keys())}")
    
    async def execute_workflow(self, workflow_type: WorkflowType, input_data: Dict[str, Any]) -> WorkflowResult:
        """Execute a complete workflow"""
        
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        start_time = datetime.utcnow()
        
        logger.info(f"Starting workflow {workflow_id} of type {workflow_type.value}")
        
        try:
            # Create workflow steps based on type
            workflow_steps = self._create_workflow_steps(workflow_type, input_data)
            
            # Execute workflow steps
            workflow_results = {}
            agent_results = {}
            reasoning_chains = {}
            errors = []
            
            for step in workflow_steps:
                try:
                    # Execute step
                    step_result = await self._execute_workflow_step(step, workflow_results)
                    
                    # Store results
                    agent_results[step.agent_name] = step_result
                    reasoning_chains[step.agent_name] = step_result.get('reasoning_chain', [])
                    workflow_results[step.step_type] = step_result
                    
                    # Update metrics
                    self.orchestration_metrics['agent_usage'][step.agent_name] += 1
                    
                    logger.info(f"Completed step {step.step_id} with agent {step.agent_name}")
                    
                except Exception as e:
                    error_msg = f"Step {step.step_id} failed: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    
                    # Check if we should continue or abort
                    if step.retry_count >= step.max_retries:
                        break
                    else:
                        step.retry_count += 1
                        continue
            
            # Calculate execution time
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Determine status
            status = 'completed' if not errors else 'completed_with_errors'
            if len(errors) > len(workflow_steps) // 2:
                status = 'failed'
            
            # Create workflow result
            result = WorkflowResult(
                workflow_id=workflow_id,
                workflow_type=workflow_type,
                status=status,
                results=workflow_results,
                agent_results=agent_results,
                reasoning_chains=reasoning_chains,
                execution_time_ms=execution_time_ms,
                errors=errors
            )
            
            # Update metrics
            self.orchestration_metrics['workflows_executed'] += 1
            self.orchestration_metrics['total_execution_time_ms'] += execution_time_ms
            self.orchestration_metrics['workflow_types'][workflow_type.value] = \
                self.orchestration_metrics['workflow_types'].get(workflow_type.value, 0) + 1
            
            if status == 'completed':
                self.orchestration_metrics['workflows_completed'] += 1
            elif status == 'failed':
                self.orchestration_metrics['workflows_failed'] += 1
            
            # Store in history
            self.workflow_history.append(result)
            
            logger.info(f"Workflow {workflow_id} completed with status: {status}")
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            result = WorkflowResult(
                workflow_id=workflow_id,
                workflow_type=workflow_type,
                status='failed',
                results={},
                agent_results={},
                reasoning_chains={},
                execution_time_ms=execution_time_ms,
                errors=[str(e)]
            )
            
            self.orchestration_metrics['workflows_executed'] += 1
            self.orchestration_metrics['workflows_failed'] += 1
            self.workflow_history.append(result)
            
            return result
    
    async def _execute_workflow_step(self, step: WorkflowStep, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        
        agent = self.agents[step.agent_name]
        
        # Prepare input data (include results from previous steps if needed)
        enhanced_input_data = step.input_data.copy()
        for dependency in step.dependencies:
            if dependency in workflow_results:
                enhanced_input_data[f"dependency_{dependency}"] = workflow_results[dependency]
        
        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                agent.process(enhanced_input_data),
                timeout=step.timeout_seconds
            )
            return result
        except asyncio.TimeoutError:
            raise Exception(f"Step {step.step_id} timed out after {step.timeout_seconds} seconds")
    
    def _create_workflow_steps(self, workflow_type: WorkflowType, input_data: Dict[str, Any]) -> List[WorkflowStep]:
        """Create workflow steps based on workflow type"""
        
        steps = []
        
        if workflow_type == WorkflowType.ANALYSIS_ONLY:
            steps.append(WorkflowStep(
                step_id="analysis_1",
                agent_name="analysis_agent",
                step_type="analysis",
                input_data=input_data,
                dependencies=[]
            ))
        
        elif workflow_type == WorkflowType.ANALYSIS_WITH_QA:
            steps.append(WorkflowStep(
                step_id="analysis_1",
                agent_name="analysis_agent",
                step_type="analysis",
                input_data=input_data,
                dependencies=[]
            ))
            steps.append(WorkflowStep(
                step_id="qa_1",
                agent_name="qa_agent",
                step_type="qa",
                input_data=input_data,
                dependencies=["analysis"]
            ))
        
        elif workflow_type == WorkflowType.QUESTION_ANSWERING:
            steps.append(WorkflowStep(
                step_id="question_1",
                agent_name="question_agent",
                step_type="question",
                input_data=input_data,
                dependencies=[]
            ))
        
        elif workflow_type == WorkflowType.COMPREHENSIVE:
            steps.append(WorkflowStep(
                step_id="analysis_1",
                agent_name="analysis_agent",
                step_type="analysis",
                input_data=input_data,
                dependencies=[]
            ))
            steps.append(WorkflowStep(
                step_id="qa_1",
                agent_name="qa_agent",
                step_type="qa",
                input_data=input_data,
                dependencies=["analysis"]
            ))
            steps.append(WorkflowStep(
                step_id="question_1",
                agent_name="question_agent",
                step_type="question",
                input_data=input_data,
                dependencies=["analysis", "qa"]
            ))
        
        return steps
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        
        agent_status = {}
        
        for agent_name, agent in self.agents.items():
            agent_status[agent_name] = {
                'agent_info': agent.get_agent_info(),
                'metrics': agent.get_metrics()
            }
            
            # Add agent-specific metrics
            if hasattr(agent, 'get_qa_metrics'):
                agent_status[agent_name]['qa_metrics'] = agent.get_qa_metrics()
            elif hasattr(agent, 'get_analysis_metrics'):
                agent_status[agent_name]['analysis_metrics'] = agent.get_analysis_metrics()
            elif hasattr(agent, 'get_question_metrics'):
                agent_status[agent_name]['question_metrics'] = agent.get_question_metrics()
        
        return agent_status
    
    async def get_reasoning_visualization(self, workflow_id: str = None) -> Dict[str, Any]:
        """Get reasoning visualization data"""
        
        if workflow_id:
            # Get specific workflow
            workflow = next((w for w in self.workflow_history if w.workflow_id == workflow_id), None)
            if not workflow:
                return {'error': f'Workflow {workflow_id} not found'}
            
            return self._format_reasoning_data(workflow)
        else:
            # Get all workflows
            all_reasoning = []
            for workflow in self.workflow_history[-10:]:  # Last 10 workflows
                formatted = self._format_reasoning_data(workflow)
                all_reasoning.append(formatted)
            
            return {
                'workflows': all_reasoning,
                'summary': self._generate_reasoning_summary()
            }
    
    def _format_reasoning_data(self, workflow: WorkflowResult) -> Dict[str, Any]:
        """Format reasoning data for visualization"""
        
        formatted_data = {
            'workflow_id': workflow.workflow_id,
            'workflow_type': workflow.workflow_type.value,
            'status': workflow.status,
            'execution_time_ms': workflow.execution_time_ms,
            'agents': []
        }
        
        for agent_name, reasoning_chain in workflow.reasoning_chains.items():
            agent_data = {
                'agent_name': agent_name,
                'reasoning_steps': reasoning_chain,
                'step_count': len(reasoning_chain),
                'avg_confidence': 0.0,
                'total_duration_ms': 0
            }
            
            if reasoning_chain:
                confidences = [step.get('confidence', 0) for step in reasoning_chain]
                durations = [step.get('duration_ms', 0) for step in reasoning_chain]
                
                agent_data['avg_confidence'] = sum(confidences) / len(confidences)
                agent_data['total_duration_ms'] = sum(durations)
            
            formatted_data['agents'].append(agent_data)
        
        return formatted_data
    
    def _generate_reasoning_summary(self) -> Dict[str, Any]:
        """Generate summary of reasoning across all workflows"""
        
        if not self.workflow_history:
            return {'message': 'No workflows executed yet'}
        
        total_workflows = len(self.workflow_history)
        total_reasoning_steps = 0
        total_agents_used = set()
        
        for workflow in self.workflow_history:
            for agent_name in workflow.reasoning_chains.keys():
                total_agents_used.add(agent_name)
                total_reasoning_steps += len(workflow.reasoning_chains[agent_name])
        
        avg_execution_time = sum(w.execution_time_ms for w in self.workflow_history) / total_workflows
        
        return {
            'total_workflows': total_workflows,
            'total_reasoning_steps': total_reasoning_steps,
            'unique_agents_used': len(total_agents_used),
            'avg_execution_time_ms': avg_execution_time,
            'agents_used': list(total_agents_used)
        }
    
    async def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration metrics"""
        
        return {
            'orchestrator_id': self.orchestrator_id,
            'orchestration_metrics': self.orchestration_metrics,
            'agent_status': await self.get_agent_status(),
            'active_workflows': len(self.active_workflows),
            'workflow_history_count': len(self.workflow_history),
            'avg_workflow_execution_time': (
                self.orchestration_metrics['total_execution_time_ms'] / 
                max(self.orchestration_metrics['workflows_executed'], 1)
            )
        }
    
    async def clear_agent_memory(self, agent_name: str = None):
        """Clear memory for specific agent or all agents"""
        
        if agent_name:
            if agent_name in self.agents:
                self.agents[agent_name].clear_reasoning_chain()
                logger.info(f"Cleared memory for agent: {agent_name}")
            else:
                raise ValueError(f"Agent {agent_name} not found")
        else:
            for agent in self.agents.values():
                agent.clear_reasoning_chain()
            logger.info("Cleared memory for all agents")
    
    async def reset_metrics(self):
        """Reset all orchestration metrics"""
        
        self.orchestration_metrics = {
            'workflows_executed': 0,
            'workflows_completed': 0,
            'workflows_failed': 0,
            'total_execution_time_ms': 0,
            'agent_usage': {agent_name: 0 for agent_name in self.agents.keys()},
            'workflow_types': {}
        }
        
        # Reset agent metrics
        for agent in self.agents.values():
            agent.metrics = {
                'tasks_completed': 0,
                'tasks_failed': 0,
                'total_reasoning_steps': 0,
                'avg_confidence': 0.0,
                'total_processing_time_ms': 0
            }
        
        logger.info("Reset all orchestration and agent metrics")
    
    def get_available_workflows(self) -> Dict[str, Any]:
        """Get information about available workflow types"""
        
        return {
            'workflow_types': {
                'analysis_only': {
                    'description': 'Run analysis agent only',
                    'agents_used': ['analysis_agent'],
                    'estimated_time_ms': 30000
                },
                'analysis_with_qa': {
                    'description': 'Run analysis followed by quality assurance',
                    'agents_used': ['analysis_agent', 'qa_agent'],
                    'estimated_time_ms': 45000
                },
                'question_answering': {
                    'description': 'Answer natural language questions',
                    'agents_used': ['question_agent'],
                    'estimated_time_ms': 15000
                },
                'comprehensive': {
                    'description': 'Run full analysis, QA, and question answering',
                    'agents_used': ['analysis_agent', 'qa_agent', 'question_agent'],
                    'estimated_time_ms': 60000
                }
            },
            'available_agents': list(self.agents.keys())
        }
