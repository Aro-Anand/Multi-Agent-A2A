"""
CrewAI Question Generator Agent

This module contains the actual CrewAI agent logic for generating
educational questions.

Components:
- QuestionGeneratorAgent: Main agent class
- CrewAI Agent: Defines role, goal, and backstory
- CrewAI Task: Defines the question generation task
- CrewAI Crew: Orchestrates agent and task execution
"""

import os
from crewai import Agent, Task, Crew, Process


class QuestionGeneratorAgent:
    """
    Question Generator using CrewAI framework.
    
    This class wraps CrewAI's Agent, Task, and Crew components
    to generate educational questions from a given topic.
    
    Architecture:
    ------------
    CrewAI uses a role-based multi-agent approach:
    
    1. Agent: Has a role, goal, and backstory
       - Think of it as a person with expertise
       - The role defines who they are
       - The goal defines what they want to achieve
       - The backstory provides context
    
    2. Task: Defines what needs to be done
       - Description: Clear instructions
       - Agent: Who should do it
       - Expected output: What format to return
    
    3. Crew: Orchestrates agents and tasks
       - Assigns tasks to agents
       - Manages execution flow
       - Returns final results
    
    Attributes:
        api_key (str): Google API key for Gemini
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the question generator.
        
        Args:
            api_key: Google API key for Gemini access
        """
        self.api_key = api_key
        os.environ['GOOGLE_API_KEY'] = api_key
    
    def _create_agent(self) -> Agent:
        """
        Create the CrewAI agent with specific role and capabilities.
        
        The agent is configured as an educational expert who specializes
        in creating thought-provoking questions.
        
        Returns:
            Agent: Configured CrewAI agent
        """
        agent = Agent(
            role='Educational Question Generator',
            
            goal='Generate insightful and diverse questions to help learners understand topics deeply',
            
            backstory="""You are an expert educator with 20 years of experience 
            creating study materials. You excel at crafting questions that:
            
            - Cover different cognitive levels (remember, understand, apply, analyze)
            - Address multiple aspects of a topic
            - Encourage critical thinking
            - Are clear and unambiguous
            - Progressively increase in difficulty
            
            You understand that good questions are essential for learning and
            take pride in generating questions that truly help students master
            the subject matter.""",
            
            verbose=True,  # Enable detailed logging
            
            # Use Google Gemini via CrewAI
            # Format: "provider/model-name"
            llm='gemini-2.0-flash',
            
            # Allow agent to delegate tasks if needed
            allow_delegation=False
        )
        
        return agent
    
    def _create_task(self, agent: Agent, topic: str) -> Task:
        """
        Create a task for generating questions on a given topic.
        
        The task provides clear instructions to the agent about:
        - What to generate (5 questions)
        - How to generate them (diverse, different difficulty levels)
        - What format to use (numbered list)
        
        Args:
            agent: The CrewAI agent that will execute this task
            topic: The subject to generate questions about
            
        Returns:
            Task: Configured CrewAI task
        """
        task = Task(
            description=f"""Generate 5 educational questions about the topic: "{topic}"
            
            Requirements:
            1. Create exactly 5 questions
            2. Cover different aspects of the topic
            3. Include a mix of difficulty levels:
               - 2 basic questions (recall, definition)
               - 2 intermediate questions (application, comparison)
               - 1 advanced question (analysis, synthesis)
            4. Make questions clear and specific
            5. Ensure questions encourage critical thinking
            
            Format your output as a numbered list:
            1. [First question]
            2. [Second question]
            3. [Third question]
            4. [Fourth question]
            5. [Fifth question]
            
            Do NOT include answers, explanations, or any other text.
            Just return the numbered questions.""",
            
            agent=agent,  # Assign this task to the agent
            
            expected_output="A numbered list of exactly 5 questions, one per line",
            
            # Optional: Add output file if you want to save results
            # output_file=f"questions_{topic.replace(' ', '_')}.txt"
        )
        
        return task
    
    def _create_crew(self, agent: Agent, task: Task) -> Crew:
        """
        Create a crew to orchestrate the agent and task.
        
        The crew manages:
        - Which agents are involved
        - Which tasks need to be completed
        - In what order (process)
        - How to handle results
        
        Args:
            agent: The agent(s) to include in the crew
            task: The task(s) to execute
            
        Returns:
            Crew: Configured CrewAI crew
        """
        crew = Crew(
            agents=[agent],  # Can have multiple agents
            
            tasks=[task],    # Can have multiple tasks
            
            # Process defines execution order:
            # - sequential: Tasks run one after another
            # - hierarchical: Manager agent delegates to workers
            process=Process.sequential,
            
            verbose=True,  # Enable detailed logging
            
            # Optional configurations:
            # manager_llm='gemini/gemini-2.0-flash-exp',  # For hierarchical
            # max_rpm=60,  # Rate limit (requests per minute)
            # share_crew=False,  # Whether agents share context
        )
        
        return crew
    
    async def generate_questions(self, topic: str) -> str:
        """
        Generate educational questions for a given topic.
        
        This is the main public method that orchestrates the entire
        question generation process.
        
        Flow:
        ----
        1. Create CrewAI Agent (educational expert)
        2. Create Task (generate 5 questions)
        3. Create Crew (orchestrate execution)
        4. Execute Crew (calls Gemini via CrewAI)
        5. Return results
        
        Args:
            topic: The subject to generate questions about
            
        Returns:
            str: Numbered list of 5 questions
            
        Example:
            >>> agent = QuestionGeneratorAgent(api_key)
            >>> questions = await agent.generate_questions("Machine Learning")
            >>> print(questions)
            1. What is the difference between supervised and unsupervised learning?
            2. How does gradient descent optimize neural network weights?
            ...
        """
        print(f"[Question Generator] Creating agent for topic: {topic}")
        
        # Step 1: Create the agent
        agent = self._create_agent()
        
        # Step 2: Create the task
        task = self._create_task(agent, topic)
        
        # Step 3: Create the crew
        crew = self._create_crew(agent, task)
        
        print(f"[Question Generator] Executing crew...")
        
        # Step 4: Execute the crew
        # kickoff() starts the execution and waits for completion
        # It internally:
        # - Sends the task to the agent
        # - Agent formulates a prompt for Gemini
        # - Calls Gemini API
        # - Processes the response
        # - Returns the result
        result = crew.kickoff()
        
        # Step 5: Extract and return the result
        # CrewAI returns a CrewOutput object
        # We convert it to string to get the actual questions
        questions = str(result)
        
        print(f"[Question Generator] Generation complete")
        print(f"[Question Generator] Questions:\n{questions}")
        
        return questions


# Example usage (for testing)
if __name__ == "__main__":
    import asyncio
    
    # Load API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        exit(1)
    
    # Create agent and generate questions
    agent = QuestionGeneratorAgent(api_key)
    
    async def test():
        questions = await agent.generate_questions("Quantum Computing")
        print("\n" + "="*80)
        print("GENERATED QUESTIONS:")
        print("="*80)
        print(questions)
    
    asyncio.run(test())