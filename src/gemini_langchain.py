"""Multi-provider LangChain-compatible LLM wrapper for agent.

Supports: Gemini (default), Groq (free tier), Ollama (local), OpenAI
Set AGENT_LLM_PROVIDER in .env to switch providers.
"""

from __future__ import annotations
import os
from typing import Any, List, Optional
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from src.gemini_client import generate_text


class AgentLLM(BaseLLM):
    """Multi-provider LLM wrapper - drop-in for ChatOpenAI.invoke().
    
    Providers:
    - 'gemini' (default): Uses Gemini API
    - 'groq': Free tier (20 req/min, 1M tokens/day) - uses Llama/Mixtral
    - 'ollama': Local models (completely free) - requires Ollama running
    - 'openai': OpenAI compatible endpoint
    
    Set via AGENT_LLM_PROVIDER env var.
    """
    provider: str = "gemini"
    model: Optional[str] = None
    temperature: float = 0.2

    def __init__(self, model: str | None = None, temperature: float = 0.2, **kwargs: Any):
        super().__init__(model=model, temperature=temperature, **kwargs)
        self.provider = os.environ.get("AGENT_LLM_PROVIDER", "gemini").lower()

    @property
    def _llm_type(self) -> str:
        return "agent_llm"

    def _get_groq_client(self):
        """Lazy-load Groq client."""
        if "_groq_client" not in self.__dict__ or self.__dict__["_groq_client"] is None:
            try:
                from groq import Groq
                api_key = os.environ.get("GROQ_API_KEY", "")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not set")
                self.__dict__["_groq_client"] = Groq(api_key=api_key)
            except ImportError:
                raise ImportError("groq package not installed. Run: pip install groq")
        return self.__dict__["_groq_client"]

    def _call_groq(self, prompt: str) -> str:
        """Call Groq API (free tier available)."""
        client = self._get_groq_client()
        model = self.model
        if not model or "gemini" in model.lower() or "gpt" in model.lower() or "llama3-8b" in model.lower() or "llama-3.1-8b" in model.lower():
            model = "llama-3.3-70b-versatile"
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a drone security analyst. Answer concisely and factually."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama instance (completely free, runs locally)."""
        import requests
        
        model = self.model or "llama3.1"  # Default local model
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": "You are a drone security analyst. Answer concisely and factually.",
                "temperature": self.temperature,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI compatible API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        model = self.model or "gpt-4o-mini"  # Cheapest option
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a drone security analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini (default)."""
        return generate_text(
            prompt,
            model=self.model,
            temperature=self.temperature,
        )

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            if self.provider == "groq":
                text = self._call_groq(prompt)
            elif self.provider == "ollama":
                text = self._call_ollama(prompt)
            elif self.provider == "openai":
                text = self._call_openai(prompt)
            else:  # gemini (default)
                text = self._call_gemini(prompt)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    def invoke(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any):
        """Invoke LLM based on configured provider."""
        if self.provider == "groq":
            text = self._call_groq(prompt)
        elif self.provider == "ollama":
            text = self._call_ollama(prompt)
        elif self.provider == "openai":
            text = self._call_openai(prompt)
        else:  # gemini (default)
            text = self._call_gemini(prompt)

        class _Response:
            def __init__(self, content: str):
                self.content = content

        return _Response(text)


# Backward compatibility alias
GeminiLangChain = AgentLLM
