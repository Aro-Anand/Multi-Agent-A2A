"""
Client Agent - Workflow Orchestrator

This module implements the client-side orchestration logic using
the A2A protocol to coordinate remote agents.

The client:
1. Discovers remote agents via Agent Cards (A2ACardResolver)
2. Calls CrewAI agent to generate questions
3. Calls LangGraph agent to convert to LaTeX
4. Returns the final result
"""

import asyncio
import logging
from typing import Optional

from .a2a_tools_ import A2ARemoteAgentTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_workflow(
    topic: str,
    api_key: str,
    crewai_url: str = "http://localhost:8001",
    langgraph_url: str = "http://localhost:8002",
    verbose: bool = False
) -> str:
    """
    Run the complete multi-agent workflow.
    
    This function orchestrates the interaction between two remote A2A agents:
    1. CrewAI Question Generator (generates questions)
    2. LangGraph LaTeX Converter (converts to LaTeX)
    
    Workflow:
    --------
    User Topic → [Client] → CrewAI Agent → Questions
                              ↓
                         LangGraph Agent → LaTeX Code
    
    A2A Protocol Flow:
    -----------------
    For each agent:
    1. Client discovers agent via A2ACardResolver
       └─> GET /.well-known/agent.json
    2. Client creates A2AClient with agent card
    3. Client sends message (POST /)
       └─> Body: {message: {role, parts, message_id}}
    4. Agent processes via AgentExecutor
       └─> execute() → CrewAI/LangGraph logic → Gemini API
    5. Agent returns response with task and artifacts
       └─> {result: {id, state, artifacts: [{parts: [{text}]}]}}
    6. Client extracts result from artifacts
    
    Args:
        topic: The subject to generate questions about
        api_key: Google API key (currently unused by client, kept for compatibility)
        crewai_url: Base URL of CrewAI agent server
        langgraph_url: Base URL of LangGraph agent server
        verbose: Enable detailed logging
        
    Returns:
        str: Final LaTeX code
        
    Raises:
        Exception: If any agent communication fails
    """
    
    def log(message: str):
        """Helper for verbose logging."""
        if verbose:
            logger.info(f"[Client] {message}")
    
    logger.info("[Client] Starting workflow...")
    log(f"Topic: {topic}")
    log(f"CrewAI URL: {crewai_url}")
    log(f"LangGraph URL: {langgraph_url}")
    
    # ========================================================================
    # STEP 1: Create tools for remote agents
    # ========================================================================
    logger.info("\n[Client] Step 1: Setting up remote agent tools...")
    
    # Tool for CrewAI Question Generator
    log("Creating tool for CrewAI agent...")
    question_generator = A2ARemoteAgentTool(
        base_url=crewai_url,
        agent_name="generate_questions",
        description="Generates educational questions about a given topic"
    )
    
    # Tool for LangGraph LaTeX Converter
    log("Creating tool for LangGraph agent...")
    latex_converter = A2ARemoteAgentTool(
        base_url=langgraph_url,
        agent_name="convert_to_latex",
        description="Converts text questions to LaTeX formatted code"
    )
    
    logger.info("[Client] ✓ Remote agent tools ready")
    
    # ========================================================================
    # STEP 2: Call CrewAI agent to generate questions
    # ========================================================================
    logger.info(f"\n[Client] Step 2: Generating questions about '{topic}'...")
    log("Discovering CrewAI agent via A2ACardResolver...")
    log(f"Agent Card URL: {crewai_url}/.well-known/agent.json")
    
    try:
        log("Sending topic to CrewAI agent...")
        questions = await question_generator.run(topic)
        
        log(f"Received {len(questions.split(chr(10)))} lines of questions")
        logger.info("[Client] ✓ Questions generated successfully")
        
        if verbose:
            print("\n" + "-"*80)
            print("Generated Questions:")
            print("-"*80)
            print(questions)
            print("-"*80 + "\n")
        else:
            # Show preview
            preview = questions[:150]
            if len(questions) > 150:
                preview += "..."
            logger.info(f"[Client] Preview: {preview}")
        
    except Exception as e:
        logger.error(f"[Client] ✗ Failed to generate questions: {e}")
        raise
    
    # ========================================================================
    # STEP 3: Call LangGraph agent to convert to LaTeX
    # ========================================================================
    logger.info(f"\n[Client] Step 3: Converting questions to LaTeX...")
    log("Discovering LangGraph agent via A2ACardResolver...")
    log(f"Agent Card URL: {langgraph_url}/.well-known/agent.json")
    
    try:
        log("Sending questions to LangGraph agent...")
        latex_code = await latex_converter.run(questions)
        
        log(f"Received {len(latex_code)} characters of LaTeX code")
        logger.info("[Client] ✓ LaTeX conversion successful")
        
        if verbose:
            print("\n" + "-"*80)
            print("Generated LaTeX Code:")
            print("-"*80)
            print(latex_code)
            print("-"*80 + "\n")
        else:
            # Show preview
            preview = latex_code[:200]
            if len(latex_code) > 200:
                preview += "..."
            logger.info(f"[Client] Preview: {preview}")
        
    except Exception as e:
        logger.error(f"[Client] ✗ Failed to convert to LaTeX: {e}")
        raise
    
    # ========================================================================
    # STEP 4: Return final result
    # ========================================================================
    logger.info("\n[Client] Step 4: Workflow complete!")
    log(f"Total LaTeX code: {len(latex_code)} characters")
    
    return latex_code


# Example usage for testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        exit(1)
    
    # Run workflow
    result = asyncio.run(
        run_workflow(
            topic="Artificial Intelligence",
            api_key=api_key,
            verbose=True
        )
    )
    
    print("\n" + "="*80)
    print("FINAL RESULT:")
    print("="*80)
    print(result)