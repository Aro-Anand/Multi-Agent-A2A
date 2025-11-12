"""
CrewAI Question Generator - Agent Executor

This module implements the AgentExecutor interface from A2A SDK.
It bridges the A2A protocol with CrewAI agent logic.

Key Components:
- AgentExecutor: Base class from a2a.server.agent_execution
- RequestContext: Contains incoming message and metadata
- EventQueue: Used to send events and updates back to client
- TaskUpdater: Helper for updating task status
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

from agent import QuestionGeneratorAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrewAIQuestionGeneratorExecutor(AgentExecutor):
    """
    AgentExecutor implementation for CrewAI-based question generation.
    
    This class bridges the A2A protocol with CrewAI agent logic:
    1. Receives A2A messages via execute()
    2. Extracts user prompt from message
    3. Calls CrewAI agent to generate questions
    4. Sends task updates via EventQueue
    5. Returns final result with task completion
    
    Flow:
    -----
    A2A Server → execute() → Extract input → Call CrewAI Agent
         → Generate questions → Update task → Complete task
    
    Attributes:
        agent (QuestionGeneratorAgent): The CrewAI agent instance
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the executor with API key.
        
        Args:
            api_key: Google API key for accessing Gemini models
        """
        super().__init__()
        self.agent = QuestionGeneratorAgent(api_key)
        logger.info("[CrewAI Executor] Initialized with Gemini")
    
    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute agent task when a message is received.
        
        This is the main method called by A2A server when a client
        sends a message to this agent.
        
        Input Structure:
        ---------------
        context: RequestContext
            └── message: Message
                ├── role: "user"
                ├── parts: List[Part]
                │   └── [0]: TextPart
                │       └── text: str (the user's prompt)
                ├── message_id: str
                ├── task_id: Optional[str] (for multi-turn)
                └── context_id: Optional[str] (for multi-turn)
        
        Processing Flow:
        ---------------
        1. Validate request (check if text input is provided)
        2. Extract user query from message parts
        3. Create or retrieve task
        4. Initialize TaskUpdater for status updates
        5. Call CrewAI agent to generate questions
        6. Update task status to "working"
        7. Get final result
        8. Add result as artifact
        9. Complete task
        
        Output Structure:
        ----------------
        Task updates sent via EventQueue:
        - TaskState.working: Processing in progress
        - Artifact: Final result (questions)
        - TaskState.completed: Task finished
        
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
                logger.error("[CrewAI Executor] Request validation failed")
                raise ServerError(error=InvalidParamsError())
            
            # Step 2: Extract user query
            # context.get_user_input() extracts text from message parts
            query = context.get_user_input()
            
            logger.info(f"\n[CrewAI Executor] Received request")
            logger.info(f"[CrewAI Executor] Query: {query}")
            logger.info(f"[CrewAI Executor] Message ID: {context.message.message_id}")
            
            # Step 3: Create or retrieve task
            # Tasks track the state of agent execution
            task = context.current_task
            if not task:
                # First message in conversation - create new task
                task = new_task(context.message)
                await event_queue.enqueue_event(task)
                logger.info(f"[CrewAI Executor] Created new task: {task.id}")
            else:
                logger.info(f"[CrewAI Executor] Continuing task: {task.id}")
            
            # Step 4: Initialize TaskUpdater
            # This helper manages task status updates
            updater = TaskUpdater(
                event_queue=event_queue,
                task_id=task.id,
                context_id=task.context_id
            )
            
            # Step 5: Update status to "working"
            logger.info("[CrewAI Executor] Generating questions...")
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    'Generating educational questions...',
                    task.context_id,
                    task.id,
                )
            )
            
            # Step 6: Call CrewAI agent to generate questions
            # This calls the CrewAI agent which uses Gemini
            questions = await self.agent.generate_questions(query)
            
            logger.info(f"[CrewAI Executor] Generated {len(questions.split(chr(10)))} questions")
            logger.info(f"[CrewAI Executor] Preview: {questions[:100]}...")
            
            # Step 7: Add result as artifact
            # Artifacts are the final outputs of the task
            await updater.add_artifact(
                [Part(root=TextPart(text=questions))],
                name='generated_questions'
            )
            
            # Step 8: Complete the task
            await updater.complete()
            
            logger.info(f"[CrewAI Executor] Task completed successfully\n")
            
        except ServerError:
            # Re-raise ServerError as-is
            raise
        except Exception as e:
            # Wrap other exceptions in ServerError
            logger.error(f'[CrewAI Executor] Error during execution: {e}', exc_info=True)
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
                logger.warning("[CrewAI Executor] Empty user input")
                return True
            return False
        except Exception as e:
            logger.error(f"[CrewAI Executor] Validation error: {e}")
            return True
    
    @override
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """
        Handle task cancellation.
        
        Called when a client wants to cancel an ongoing task.
        For this agent, tasks are typically short-lived, so
        we raise UnsupportedOperationError.
        
        Args:
            context: Contains cancellation request details
            event_queue: Queue for sending cancellation acknowledgment
            
        Raises:
            ServerError: Always raises with UnsupportedOperationError
        """
        logger.warning(f"[CrewAI Executor] Cancellation requested for task: {context.current_task.id if context.current_task else 'unknown'}")
        raise ServerError(error=UnsupportedOperationError())