"""
A2A Remote Agent Tools

This module provides tools for interacting with remote A2A agents using
the official A2A SDK.

Key Components:
- A2ACardResolver: For discovering agents via Agent Cards
- A2AClient: For sending messages to agents
- Message handling: Following A2A protocol format
"""

import logging
from typing import Optional
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class A2ARemoteAgentTool:
    """
    Tool for interacting with remote A2A agents.
    
    This class provides a simple interface for:
    1. Service discovery (fetching Agent Card via A2ACardResolver)
    2. Sending messages to remote agents (via A2AClient)
    3. Receiving and parsing responses
    
    A2A Protocol Flow:
    -----------------
    1. Discovery Phase:
       A2ACardResolver → GET /.well-known/agent.json
       → Returns Agent Card (metadata about agent)
       → Client parses card to understand capabilities
    
    2. Message Exchange:
       A2AClient → POST / with JSON-RPC message
       → Agent processes via AgentExecutor
       → Agent returns response with task updates
       → Client extracts result from artifacts
    
    Example:
        >>> tool = A2ARemoteAgentTool(
        ...     base_url="http://localhost:8001",
        ...     agent_name="question_generator"
        ... )
        >>> result = await tool.run("Machine Learning")
        >>> print(result)
    """
    
    def __init__(
        self,
        base_url: str,
        agent_name: str,
        description: Optional[str] = None
    ):
        """
        Initialize the A2A remote agent tool.
        
        Args:
            base_url: Base URL of the remote agent (e.g., "http://localhost:8001")
            agent_name: Name identifier for the agent
            description: Optional description of what the agent does
        """
        self.base_url = base_url
        self.agent_name = agent_name
        self.description = description or f"Remote A2A agent at {base_url}"
        
        logger.info(f"[A2A Tool] Initialized: {agent_name}")
        logger.info(f"[A2A Tool] Base URL: {base_url}")
    
    async def run(
        self,
        input_text: str,
        timeout: float = 120.0,
        task_id: Optional[str] = None,
        context_id: Optional[str] = None
    ) -> str:
        """
        Execute the remote agent with the given input.
        
        This method:
        1. Creates HTTP client
        2. Discovers agent via Agent Card (using A2ACardResolver)
        3. Creates A2AClient
        4. Creates A2A message following protocol format
        5. Sends message to agent
        6. Waits for response
        7. Extracts and returns result from artifacts
        
        Detailed Flow:
        -------------
        
        Step 1: Create HTTP Client
        ---------------------------
        async with httpx.AsyncClient(timeout=timeout)
            └── Manages HTTP connections
            └── Handles timeouts
            └── Reuses connections
        
        Step 2: Service Discovery
        ------------------------
        A2ACardResolver(httpx_client, base_url)
            ├─> GET /.well-known/agent.json
            │
            ├─> Parse Agent Card JSON:
            │   {
            │     "name": "Agent Name",
            │     "url": "http://...",
            │     "skills": [...],
            │     "capabilities": {...}
            │   }
            │
            └─> Return AgentCard object
        
        Step 3: Create A2A Client
        -------------------------
        A2AClient(httpx_client, agent_card)
            └── Configured with agent's capabilities
            └── Ready to send messages
        
        Step 4: Create A2A Message
        --------------------------
        send_message_payload = {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": input_text
                    }
                ],
                "message_id": uuid4().hex,
                "task_id": task_id (optional),
                "context_id": context_id (optional)
            }
        }
        
        Step 5: Send Message
        -------------------
        await client.send_message(request)
            │
            ├─> POST http://base_url/
            │   Headers: Content-Type: application/json
            │   Body: JSON-RPC 2.0 message
            │
            ├─> A2A Server receives request
            │
            ├─> Routes to AgentExecutor.execute()
            │
            ├─> Agent processes (CrewAI, LangGraph, etc.)
            │
            └─> Returns response with task and artifacts
        
        Step 6: Extract Result
        ---------------------
        response.root.result
            ├─> artifacts: List[Artifact]
            │   └─> [0].parts[0].text
            └─> Extract text from first artifact
        
        Args:
            input_text: Text to send to the remote agent
            timeout: Request timeout in seconds (default: 120s)
            task_id: Optional task ID for multi-turn conversations
            context_id: Optional context ID for multi-turn conversations
            
        Returns:
            str: Response text from the remote agent's artifacts
            
        Raises:
            httpx.HTTPError: If network communication fails
            ValueError: If response format is invalid
            Exception: If agent returns an error
        """
        
        # Create HTTP client with timeout
        async with httpx.AsyncClient(timeout=timeout) as httpx_client:
            
            try:
                # ============================================================
                # STEP 1: Service Discovery using A2ACardResolver
                # ============================================================
                logger.info(f"[A2A Tool] Discovering agent: {self.agent_name}")
                logger.info(f"[A2A Tool] Fetching Agent Card from {self.base_url}")
                
                # Initialize A2ACardResolver
                resolver = A2ACardResolver(
                    httpx_client=httpx_client,
                    base_url=self.base_url,
                    # Uses default agent_card_path: /.well-known/agent.json
                )
                
                # Fetch the Agent Card
                agent_card: AgentCard = await resolver.get_agent_card()
                
                logger.info(f"[A2A Tool] ✓ Agent Card fetched: {agent_card.name}")
                logger.info(f"[A2A Tool]   Version: {agent_card.version}")
                logger.info(f"[A2A Tool]   Skills: {[s.name for s in agent_card.skills]}")
                
                # ============================================================
                # STEP 2: Initialize A2A Client
                # ============================================================
                client = A2AClient(
                    httpx_client=httpx_client,
                    agent_card=agent_card
                )
                
                logger.info(f"[A2A Tool] ✓ A2A Client initialized")
                
                # ============================================================
                # STEP 3: Create A2A Message
                # ============================================================
                logger.info(f"[A2A Tool] Creating message...")
                
                # Generate unique message ID
                message_id = uuid4().hex
                
                # Build message payload following A2A protocol
                # Note: Use "kind" instead of "type" for parts
                message_payload = {
                    'message': {
                        'role': 'user',
                        'parts': [
                            {
                                'kind': 'text',  # A2A SDK uses 'kind' not 'type'
                                'text': input_text
                            }
                        ],
                        'message_id': message_id
                    }
                }
                
                # Add task_id and context_id if provided (for multi-turn)
                if task_id:
                    message_payload['message']['task_id'] = task_id
                if context_id:
                    message_payload['message']['context_id'] = context_id
                
                logger.info(f"[A2A Tool] Message ID: {message_id}")
                logger.info(f"[A2A Tool] Input length: {len(input_text)} characters")
                if task_id:
                    logger.info(f"[A2A Tool] Task ID: {task_id}")
                if context_id:
                    logger.info(f"[A2A Tool] Context ID: {context_id}")
                
                # Wrap in SendMessageRequest
                request = SendMessageRequest(
                    id=str(uuid4()),  # Request ID
                    params=MessageSendParams(**message_payload)
                )
                
                # ============================================================
                # STEP 4: Send Message and Wait for Response
                # ============================================================
                logger.info(f"[A2A Tool] Sending message to {self.base_url}...")
                logger.info(f"[A2A Tool] Waiting for response (timeout: {timeout}s)...")
                
                # This is the actual HTTP POST request
                # It blocks until the agent responds or timeout occurs
                response = await client.send_message(request)
                
                logger.info(f"[A2A Tool] ✓ Response received")
                
                # ============================================================
                # STEP 5: Extract Result from Response
                # ============================================================
                logger.info(f"[A2A Tool] Parsing response...")
                
                # Response structure:
                # response.root
                #   └── result: TaskResult
                #       ├── id: task_id
                #       ├── context_id: context_id
                #       ├── state: TaskState
                #       └── artifacts: List[Artifact]
                #           └── [0]: Artifact
                #               └── parts: List[Part]
                #                   └── [0]: Part
                #                       └── root: TextPart
                #                           └── text: str
                
                result_text = ""
                
                # Check if response has result
                if response.root and response.root.result:
                    task_result = response.root.result
                    
                    # Log task info
                    logger.info(f"[A2A Tool] Task ID: {task_result.id}")
                    
                    # Check for state attribute (may not exist in all response types)
                    if hasattr(task_result, 'state'):
                        logger.info(f"[A2A Tool] Task State: {task_result.state}")
                    
                    # Extract text from artifacts
                    if hasattr(task_result, 'artifacts') and task_result.artifacts and len(task_result.artifacts) > 0:
                        artifact = task_result.artifacts[0]
                        
                        if hasattr(artifact, 'parts') and artifact.parts and len(artifact.parts) > 0:
                            part = artifact.parts[0]
                            
                            # Access the text from Part.root.text
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                result_text = part.root.text
                    
                    # If no artifacts, check message parts (fallback)
                    if not result_text and hasattr(task_result, 'message') and task_result.message:
                        if hasattr(task_result.message, 'parts') and task_result.message.parts and len(task_result.message.parts) > 0:
                            part = task_result.message.parts[0]
                            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                                result_text = part.root.text
                
                if not result_text:
                    raise ValueError("No text content in response artifacts or message")
                
                logger.info(f"[A2A Tool] ✓ Result extracted ({len(result_text)} characters)")
                
                return result_text
                
            except httpx.TimeoutException as e:
                error_msg = f"Request timeout after {timeout}s"
                logger.error(f"[A2A Tool] ✗ {error_msg}")
                raise Exception(error_msg) from e
            
            except httpx.HTTPError as e:
                error_msg = f"Network error: {str(e)}"
                logger.error(f"[A2A Tool] ✗ {error_msg}")
                raise Exception(error_msg) from e
            
            except Exception as e:
                error_msg = f"A2A communication error: {str(e)}"
                logger.error(f"[A2A Tool] ✗ {error_msg}")
                raise Exception(error_msg) from e


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Test with CrewAI agent
        crewai_tool = A2ARemoteAgentTool(
            base_url="http://localhost:8001",
            agent_name="crewai_question_generator",
            description="Generates questions"
        )
        
        try:
            result = await crewai_tool.run("Machine Learning")
            print("\n" + "="*80)
            print("RESULT:")
            print("="*80)
            print(result)
        except Exception as e:
            print(f"Error: {e}")
    
    # Run test
    asyncio.run(test())