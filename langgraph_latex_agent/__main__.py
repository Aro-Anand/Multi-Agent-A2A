"""
LangGraph LaTeX Converter Agent - A2A Server Entry Point

This module starts an A2A-compliant server using the official SDK structure.

Usage:
    python -m langgraph_latex_agent
    python -m langgraph_latex_agent --port 8002 --host 0.0.0.0
"""

import logging
import os
import sys

import click
import httpx
import uvicorn
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent_executor import LangGraphLatexExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option('--host', default='localhost', help='Host to bind server to')
@click.option('--port', default=8002, help='Port to run server on')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def main(host: str, port: int, debug: bool):
    """
    Starts the LangGraph LaTeX Converter A2A server.
    
    This server exposes a LangGraph agent via the A2A protocol for converting
    text questions to LaTeX formatted code.
    """
    try:
        # Set debug logging if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Validate API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise MissingAPIKeyError(
                'GOOGLE_API_KEY environment variable not set. '
                'Please set it in .env file or export it.'
            )
        
        logger.info("="*80)
        logger.info("LANGGRAPH LATEX CONVERTER - A2A SERVER")
        logger.info("="*80)
        logger.info(f"Host: {host}")
        logger.info(f"Port: {port}")
        logger.info(f"Agent Card URL: http://{host}:{port}/.well-known/agent.json")
        logger.info("="*80)
        
        # Define agent capabilities
        capabilities = AgentCapabilities(
            streaming=True,
            push_notifications=True
        )
        
        # Define agent skill
        skill = AgentSkill(
            id='convert_to_latex',
            name='LaTeX Converter',
            description='Converts text questions to professionally formatted LaTeX code with proper document structure',
            tags=['latex', 'formatting', 'academic', 'document', 'typesetting'],
            examples=[
                'Convert these questions to LaTeX: 1. What is AI? 2. How does ML work?',
                'Format in LaTeX: Question 1: Define quantum computing',
                'LaTeX document with numbered questions about physics'
            ],
        )
        
        # Create Agent Card
        agent_card = AgentCard(
            name='LangGraph LaTeX Converter',
            description=(
                'An A2A agent that converts text questions to LaTeX formatted code '
                'using LangGraph state machine and Google Gemini. Generates '
                'compile-ready LaTeX documents with proper structure and formatting.'
            ),
            url=f'http://{host}:{port}/',
            version='1.0.0',
            default_input_modes=['text', 'text/plain'],
            default_output_modes=['text', 'text/plain'],
            capabilities=capabilities,
            skills=[skill],
        )
        
        logger.info("Agent Card created:")
        logger.info(f"  - Name: {agent_card.name}")
        logger.info(f"  - Version: {agent_card.version}")
        logger.info(f"  - Skills: {[s.name for s in agent_card.skills]}")
        
        # Initialize HTTP client
        httpx_client = httpx.AsyncClient()
        
        # Setup push notification infrastructure
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )
        
        # Create request handler
        request_handler = DefaultRequestHandler(
            agent_executor=LangGraphLatexExecutor(api_key=api_key),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )
        
        logger.info("Request handler initialized")
        
        # Create A2A Starlette application
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        
        logger.info("A2A Starlette application created")
        logger.info(f"Starting server on http://{host}:{port}")
        logger.info("Press CTRL+C to stop")
        logger.info("")
        
        # Start the server
        uvicorn.run(server.build(), host=host, port=port)
        
    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()