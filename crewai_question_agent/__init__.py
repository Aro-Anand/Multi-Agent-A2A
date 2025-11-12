# =============================================================================
# crewai_question_agent/__init__.py
# =============================================================================
"""
CrewAI Question Generator Agent Package

This package provides an A2A-compliant agent that generates educational
questions using the CrewAI framework and Google Gemini.
"""

__version__ = "1.0.0"
__author__ = "Anand"

from .agent import QuestionGeneratorAgent
from .agent_executor import CrewAIQuestionGeneratorExecutor

__all__ = [
    "QuestionGeneratorAgent",
    "CrewAIQuestionGeneratorExecutor"
]


# =============================================================================
# langgraph_latex_agent/__init__.py
# =============================================================================
"""
LangGraph LaTeX Converter Agent Package

This package provides an A2A-compliant agent that converts questions to
LaTeX code using LangGraph state machine and Google Gemini.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .agent import LatexConverterAgent
from .agent_executor import LangGraphLatexExecutor

__all__ = [
    "LatexConverterAgent",
    "LangGraphLatexExecutor"
]


# =============================================================================
# client/__init__.py
# =============================================================================
"""
Client Package

This package provides the client-side orchestration for the multi-agent
A2A workflow.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .client_agent import run_workflow
from .a2a_tools import A2ARemoteAgentTool

__all__ = [
    "run_workflow",
    "A2ARemoteAgentTool"
]


# =============================================================================
# shared/__init__.py
# =============================================================================
"""
Shared Utilities Package

Common utilities and helpers used across the multi-agent system.
"""

__version__ = "1.0.0"

from .utils import *

__all__ = [
    # Add utility functions here as you create them
]