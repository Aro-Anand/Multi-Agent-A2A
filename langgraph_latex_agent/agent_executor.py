"""
LangGraph LaTeX Converter - Agent Executor

This module implements the AgentExecutor interface for LangGraph-based
LaTeX conversion using a state machine workflow.
"""

import logging
from typing import override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from agent import LatexConverterAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangGraphLatexExecutor(AgentExecutor):
    """
    AgentExecutor implementation for LangGraph-based LaTeX conversion.
    
    This class bridges the A2A protocol with LangGraph state machine:
    1. Receives A2A messages via execute()
    2. Extracts questions from message
    3. Runs LangGraph workflow to convert to LaTeX
    4. Sends task updates via EventQueue
    5. Returns final LaTeX code with task completion
    
    Workflow:
    --------
    A2A Server → execute() → Extract questions → Run LangGraph
         → Node 1: parse_questions
         → Node 2: convert_to_latex (Gemini)
         → Node 3: format_output
         → Complete task with LaTeX artifact
    
    Attributes:
        agent (LatexConverterAgent): The LangGraph agent instance
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the executor with API key.
        
        Args:
            api_key: Google API key for accessing Gemini models
        """
        super().__init__()
        self.agent = LatexConverterAgent(api_key)
        logger.info("[LangGraph Executor] Initialized with Gemini")
        logger.info("[LangGraph Executor] State graph compiled and ready")
    
    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute agent task when a message is received.
        
        Input Structure:
        ---------------
        context: RequestContext
            └── message: Message
                ├── role: "user"
                ├── parts: List[Part]
                │   └── [0]: TextPart
                │       └── text: str (questions to convert)
                ├── message_id: str
                ├── task_id: Optional[str]
                └── context_id: Optional[str]
        
        Processing Flow:
        ---------------
        1. Validate request
        2. Extract questions from message
        3. Create or retrieve task
        4. Initialize TaskUpdater
        5. Update status to "working"
        6. Run LangGraph workflow:
           - Node 1: parse_questions (extract individual questions)
           - Node 2: convert_to_latex (call Gemini)
           - Node 3: format_output (final formatting)
        7. Get LaTeX code from final state
        8. Add LaTeX code as artifact
        9. Complete task
        
        LangGraph State Flow:
        --------------------
        Initial State:
        {
            "questions": "1. Q1\n2. Q2\n...",
            "latex_code": "",
            "metadata": {}
        }
        
        After parse_questions:
        {
            "questions": "1. Q1\n2. Q2\n...",
            "latex_code": "",
            "metadata": {
                "parsed_questions": ["Q1", "Q2", ...],
                "count": 5
            }
        }
        
        After convert_to_latex:
        {
            "questions": "1. Q1\n2. Q2\n...",
            "latex_code": "\\documentclass{article}...",
            "metadata": {...}
        }
        
        After format_output (final):
        {
            "questions": "1. Q1\n2. Q2\n...",
            "latex_code": "% Generated...\n\\documentclass{article}...",
            "metadata": {...}
        }
        
        Args:
            context: Contains the incoming message and request metadata
            event_queue: Queue for sending events/responses back to client
            
        Raises:
            ServerError: If validation fails or processing error occurs
        """
        try:
            # Step 1: Validate request
            error = self._validate_request(context)
            if error:
                logger.error("[LangGraph Executor] Request validation failed")
                raise ServerError(error=InvalidParamsError())
            
            # Step 2: Extract questions
            questions_text = context.get_user_input()
            
            logger.info(f"\n[LangGraph Executor] Received request")
            logger.info(f"[LangGraph Executor] Questions preview: {questions_text[:100]}...")
            logger.info(f"[LangGraph Executor] Message ID: {context.message.message_id}")
            
            # Step 3: Create or retrieve task
            task = context.current_task
            if not task:
                task = new_task(context.message)
                await event_queue.enqueue_event(task)
                logger.info(f"[LangGraph Executor] Created new task: {task.id}")
            else:
                logger.info(f"[LangGraph Executor] Continuing task: {task.id}")
            
            # Step 4: Initialize TaskUpdater
            updater = TaskUpdater(
                event_queue=event_queue,
                task_id=task.id,
                context_id=task.context_id
            )
            
            # Step 5: Update status - parsing
            logger.info("[LangGraph Executor] Starting LaTeX conversion workflow...")
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    'Parsing questions...',
                    task.context_id,
                    task.id,
                )
            )
            
            # Step 6: Update status - converting
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    'Converting to LaTeX format...',
                    task.context_id,
                    task.id,
                )
            )
            
            # Step 7: Run LangGraph workflow
            # This executes the state machine:
            # START -> parse_questions -> convert_to_latex -> format_output -> END
            latex_code = await self.agent.convert_to_latex(questions_text)
            
            logger.info(f"[LangGraph Executor] Conversion complete")
            logger.info(f"[LangGraph Executor] LaTeX code length: {len(latex_code)} characters")
            logger.info(f"[LangGraph Executor] Preview: {latex_code[:200]}...")
            
            # Step 8: Add LaTeX code as artifact
            await updater.add_artifact(
                [Part(root=TextPart(text=latex_code))],
                name='latex_document'
            )
            
            # Step 9: Complete the task
            await updater.complete()
            
            logger.info(f"[LangGraph Executor] Task completed successfully\n")
            
        except ServerError:
            raise
        except Exception as e:
            logger.error(f'[LangGraph Executor] Error during execution: {e}', exc_info=True)
            raise ServerError(error=InternalError()) from e
    
    def _validate_request(self, context: RequestContext) -> bool:
        """
        Validate the incoming request.
        
        Checks:
        - Message has text parts
        - User input is not empty
        
        Args:
            context: Request context to validate
            
        Returns:
            bool: True if invalid, False if valid
        """
        try:
            user_input = context.get_user_input()
            if not user_input or not user_input.strip():
                logger.warning("[LangGraph Executor] Empty user input")
                return True
            return False
        except Exception as e:
            logger.error(f"[LangGraph Executor] Validation error: {e}")
            return True
    
    @override
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """
        Handle task cancellation.
        
        Args:
            context: Contains cancellation request details
            event_queue: Queue for sending cancellation acknowledgment
            
        Raises:
            ServerError: Always raises with UnsupportedOperationError
        """
        logger.warning(f"[LangGraph Executor] Cancellation requested for task: {context.current_task.id if context.current_task else 'unknown'}")
        raise ServerError(error=UnsupportedOperationError())