"""
AI Orchestration System - Multi-Agent Architecture

This package implements a comprehensive AI orchestration system with multiple specialized agents:
- QA Agent: Quality assurance and validation
- Analysis Agent: Deep video analysis
- Question Agent: Natural language queries
- Orchestration Agent: Coordination and workflow management
"""

from .orchestrator import AIOrchestrator, WorkflowType
from .qa_agent import QAAgent
from .analysis_agent import AnalysisAgent
from .question_agent import QuestionAgent
from .base_agent import BaseAgent

__all__ = [
    'AIOrchestrator',
    'WorkflowType',
    'QAAgent', 
    'AnalysisAgent',
    'QuestionAgent',
    'BaseAgent'
]
