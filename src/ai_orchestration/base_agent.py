"""
base_agent.py - Base class for all AI agents

Provides common functionality for all agents including:
- Reasoning chain management
- State tracking
- Communication interfaces
- Logging and monitoring
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import uuid

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    WAITING = "waiting"

class ReasoningStep(BaseModel):
    """Individual reasoning step"""
    step_id: str = Field(description="Unique step identifier")
    timestamp: datetime = Field(description="When step was created")
    agent_name: str = Field(description="Agent that created the step")
    step_type: str = Field(description="Type of reasoning step")
    description: str = Field(description="Step description")
    input_data: Dict[str, Any] = Field(description="Input data for step")
    reasoning: str = Field(description="Reasoning process")
    output_data: Dict[str, Any] = Field(description="Output data from step")
    confidence: float = Field(description="Confidence in step result")
    duration_ms: int = Field(description="Duration in milliseconds")
    metadata: Dict[str, Any] = Field(description="Additional metadata")

@dataclass
class AgentState:
    """Agent state tracking"""
    agent_id: str
    agent_name: str
    status: AgentStatus
    current_task: Optional[str]
    reasoning_chain: List[ReasoningStep]
    memory: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: datetime
    last_updated: datetime

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, agent_name: str, openai_api_key: str = None):
        self.agent_name = agent_name
        self.agent_id = f"{agent_name}_{uuid.uuid4().hex[:8]}"
        
        # Initialize LLM — orchestration agents use NVIDIA NIM or Groq.
        # NVIDIA NIM (Nemotron-3-Ultra-550B) is preferred for agentic reasoning.
        from src.config import settings
        import os
        
        nvidia_api_key = getattr(settings, 'NVIDIA_API_KEY', '') or os.environ.get("NVIDIA_API_KEY", "")
        nvidia_model = getattr(settings, 'NVIDIA_MODEL', '') or os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
        groq_api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        llm_provider = os.environ.get("AGENT_LLM_PROVIDER", getattr(settings, 'AGENT_LLM_PROVIDER', 'nvidia')).lower()
        
        if nvidia_api_key and llm_provider == "nvidia":
            try:
                self.llm = ChatOpenAI(
                    model=nvidia_model,
                    base_url="https://integrate.api.nvidia.com/v1",
                    openai_api_key=nvidia_api_key,
                    temperature=0.1,
                    max_retries=2
                )
                logger.info(f"[{agent_name}] Initialized using NVIDIA NIM ({nvidia_model})")
            except Exception as e:
                logger.warning(f"[{agent_name}] NVIDIA NIM init failed: {e}, falling back to Groq")
                if groq_api_key:
                    self.llm = ChatOpenAI(
                        model="llama-3.3-70b-versatile",
                        base_url="https://api.groq.com/openai/v1",
                        openai_api_key=groq_api_key,
                        temperature=0.1
                    )
                    logger.info(f"[{agent_name}] Fallback to Groq (llama-3.3-70b-versatile)")
        elif groq_api_key:
            self.llm = ChatOpenAI(
                model="llama-3.3-70b-versatile",
                base_url="https://api.groq.com/openai/v1",
                openai_api_key=groq_api_key,
                temperature=0.1
            )
            logger.info(f"[{agent_name}] Initialized using Groq (llama-3.3-70b-versatile)")
        else:
            # Fallback to OpenAI only if nothing else available
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.1,
                openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY") or "sk-proj-..."
            )
            logger.info(f"[{agent_name}] Initialized using OpenAI (fallback)")
        
        # Initialize state
        self.state = AgentState(
            agent_id=self.agent_id,
            agent_name=agent_name,
            status=AgentStatus.IDLE,
            current_task=None,
            reasoning_chain=[],
            memory={},
            metrics={},
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        # Performance metrics
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_reasoning_steps': 0,
            'avg_confidence': 0.0,
            'total_processing_time_ms': 0
        }
        
        logger.info(f"Initialized {agent_name} agent with ID: {self.agent_id}")
    
    def add_reasoning_step(
        self,
        step_type: str,
        description: str,
        input_data: Dict[str, Any],
        reasoning: str,
        output_data: Dict[str, Any],
        confidence: float,
        duration_ms: int,
        metadata: Dict[str, Any] = None
    ) -> ReasoningStep:
        """Add a reasoning step to the chain"""
        
        step = ReasoningStep(
            step_id=f"step_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.utcnow(),
            agent_name=self.agent_name,
            step_type=step_type,
            description=description,
            input_data=input_data,
            reasoning=reasoning,
            output_data=output_data,
            confidence=confidence,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        
        self.state.reasoning_chain.append(step)
        self.state.last_updated = datetime.utcnow()
        
        # Update metrics
        self.metrics['total_reasoning_steps'] += 1
        self.metrics['avg_confidence'] = (
            (self.metrics['avg_confidence'] * (self.metrics['total_reasoning_steps'] - 1) + confidence) /
            self.metrics['total_reasoning_steps']
        )
        
        logger.info(f"Added reasoning step for {self.agent_name}: {description}")
        
        return step
    
    def get_reasoning_chain(self) -> List[Dict[str, Any]]:
        """Get reasoning chain as serializable format"""
        return [
            {
                'step_id': step.step_id,
                'timestamp': step.timestamp.isoformat(),
                'agent_name': step.agent_name,
                'step_type': step.step_type,
                'description': step.description,
                'input_data': step.input_data,
                'reasoning': step.reasoning,
                'output_data': step.output_data,
                'confidence': step.confidence,
                'duration_ms': step.duration_ms,
                'metadata': step.metadata
            }
            for step in self.state.reasoning_chain
        ]
    
    def clear_reasoning_chain(self):
        """Clear the reasoning chain"""
        self.state.reasoning_chain.clear()
        self.state.last_updated = datetime.utcnow()
        logger.info(f"Cleared reasoning chain for {self.agent_name}")
    
    def update_status(self, status: AgentStatus, current_task: str = None):
        """Update agent status"""
        self.state.status = status
        self.state.current_task = current_task
        self.state.last_updated = datetime.utcnow()
        logger.info(f"Updated {self.agent_name} status to {status.value}")
    
    def store_memory(self, key: str, value: Any):
        """Store data in agent memory"""
        self.state.memory[key] = value
        self.state.last_updated = datetime.utcnow()
        logger.info(f"Stored memory for {self.agent_name}: {key}")
    
    def get_memory(self, key: str) -> Any:
        """Retrieve data from agent memory"""
        return self.state.memory.get(key)
    
    async def call_llm(self, messages: List[str], system_prompt: str = None) -> str:
        """Call LLM with messages"""
        start_time = datetime.utcnow()
        
        try:
            # Prepare messages
            langchain_messages = []
            
            if system_prompt:
                langchain_messages.append(SystemMessage(content=system_prompt))
            
            for message in messages:
                if isinstance(message, str):
                    langchain_messages.append(HumanMessage(content=message))
                else:
                    langchain_messages.append(message)
            
            # Call LLM
            response = await self.llm.ainvoke(langchain_messages)
            
            # Calculate duration
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self.metrics['total_processing_time_ms'] += duration_ms
            
            return response.content
            
        except Exception as e:
            logger.error(f"LLM call failed for {self.agent_name}: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'status': self.state.status.value,
            'current_task': self.state.current_task,
            'reasoning_steps_count': len(self.state.reasoning_chain),
            'memory_keys_count': len(self.state.memory),
            'metrics': self.metrics.copy(),
            'created_at': self.state.created_at.isoformat(),
            'last_updated': self.state.last_updated.isoformat()
        }
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get system prompt for the agent - must be implemented by subclasses"""
        pass
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'status': self.state.status.value,
            'current_task': self.state.current_task,
            'reasoning_chain': self.get_reasoning_chain(),
            'memory_keys': list(self.state.memory.keys()),
            'metrics': self.get_metrics(),
            'created_at': self.state.created_at.isoformat(),
            'last_updated': self.state.last_updated.isoformat()
        }
