"""
LangGraph LaTeX Converter Agent

This module implements a state-based workflow using LangGraph for
converting text questions to LaTeX formatted code.

Key Concepts:
- StateGraph: Defines a state machine workflow
- State: TypedDict that holds data throughout the workflow
- Nodes: Functions that process and update state
- Edges: Define transitions between nodes
"""

import os
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
import google.generativeai as genai


class LatexConverterState(TypedDict):
    """
    State definition for the LaTeX conversion workflow.
    
    LangGraph uses a state object that flows through all nodes.
    Each node can read from and write to this state.
    
    Attributes:
        questions: str - Input questions text
        latex_code: str - Output LaTeX code
        metadata: Dict - Intermediate data (parsed questions, counts, etc.)
    """
    questions: str
    latex_code: str
    metadata: Dict[str, Any]


class LatexConverterAgent:
    """
    LaTeX Converter using LangGraph state machine.
    
    This agent implements a multi-step workflow:
    1. Parse questions (extract individual questions)
    2. Convert to LaTeX (call Gemini)
    3. Format output (add metadata, final touches)
    
    Architecture:
    ------------
    LangGraph uses a state graph where:
    - Each node is a function that processes state
    - Edges define the flow between nodes
    - State is immutable - nodes return updates
    - Graph execution is deterministic
    
    Workflow:
    --------
    START
      │
      ▼
    parse_questions (Node 1)
      │ Updates: metadata.parsed_questions, metadata.count
      ▼
    convert_to_latex (Node 2)
      │ Updates: latex_code
      ▼
    format_output (Node 3)
      │ Updates: latex_code (final)
      ▼
    END
    
    Attributes:
        api_key (str): Google API key for Gemini
        model: Gemini model instance
        graph: Compiled LangGraph state machine
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the LaTeX converter.
        
        Args:
            api_key: Google API key for Gemini access
        """
        self.api_key = api_key
        os.environ['GOOGLE_API_KEY'] = api_key
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Build and compile the state graph
        self.graph = self._build_graph()
        
        print("[LaTeX Converter] Initialized")
        print("[LaTeX Converter] State graph compiled with 3 nodes")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine workflow.
        
        This creates a graph with three processing nodes:
        1. parse_questions: Extract individual questions
        2. convert_to_latex: Generate LaTeX code
        3. format_output: Final formatting
        
        Returns:
            Compiled StateGraph ready for execution
        """
        # Create the graph with our state schema
        workflow = StateGraph(LatexConverterState)
        
        # Define Node 1: Parse questions
        def parse_questions(state: LatexConverterState) -> Dict:
            """
            Parse input questions and extract individual items.
            
            Input:
            ------
            state["questions"]: str
                Raw text with questions (numbered or not)
            
            Processing:
            ----------
            - Split by newlines
            - Filter empty lines
            - Store count and parsed list
            
            Output:
            ------
            Returns updates to state:
            {
                "metadata": {
                    "parsed_questions": ["Q1", "Q2", ...],
                    "count": 5
                }
            }
            
            Args:
                state: Current state
                
            Returns:
                Dict with state updates
            """
            questions = state["questions"]
            print(f"[Node: parse_questions] Parsing input...")
            
            # Split by newlines and filter empty
            lines = questions.strip().split('\n')
            parsed = [line.strip() for line in lines if line.strip()]
            
            count = len(parsed)
            print(f"[Node: parse_questions] Found {count} questions")
            
            # Return state updates
            return {
                "metadata": {
                    "parsed_questions": parsed,
                    "count": count
                }
            }
        
        # Define Node 2: Convert to LaTeX
        def convert_to_latex(state: LatexConverterState) -> Dict:
            """
            Convert questions to LaTeX using Gemini.
            
            Input:
            ------
            state["metadata"]["parsed_questions"]: List[str]
                Individual questions to convert
            
            Processing:
            ----------
            1. Create prompt for Gemini
            2. Call Gemini API
            3. Extract LaTeX code from response
            4. Clean markdown formatting
            
            Output:
            ------
            Returns updates to state:
            {
                "latex_code": "\\documentclass{article}..."
            }
            
            Args:
                state: Current state
                
            Returns:
                Dict with state updates
            """
            parsed = state["metadata"]["parsed_questions"]
            count = state["metadata"]["count"]
            
            print(f"[Node: convert_to_latex] Converting {count} questions to LaTeX...")
            
            # Create detailed prompt for Gemini
            prompt = f"""Convert the following questions into a professional LaTeX document.

Questions:
{chr(10).join(parsed)}

Requirements:
1. Use \\documentclass{{article}}
2. Include necessary packages:
   - \\usepackage{{enumerate}} for numbered lists
   - \\usepackage{{amsmath}} for any math formatting
   - \\usepackage[margin=1in]{{geometry}} for margins
3. Use \\begin{{enumerate}} environment for the questions
4. Each question should be a \\item
5. Make the document compile-ready (complete with \\begin{{document}} and \\end{{document}})
6. Add a title like "Educational Questions"
7. Use proper LaTeX formatting and escaping

Return ONLY the LaTeX code, no explanations or markdown code blocks."""
            
            # Call Gemini
            response = self.model.generate_content(prompt)
            latex_code = response.text
            
            # Clean up markdown code blocks if present
            # Gemini sometimes wraps code in ```latex...```
            if "```latex" in latex_code:
                # Extract content between ```latex and ```
                latex_code = latex_code.split("```latex")[1].split("```")[0].strip()
            elif "```" in latex_code:
                # Extract content between ``` and ```
                latex_code = latex_code.split("```")[1].split("```")[0].strip()
            
            print(f"[Node: convert_to_latex] Conversion complete")
            print(f"[Node: convert_to_latex] Generated {len(latex_code)} characters")
            
            # Return state updates
            return {"latex_code": latex_code}
        
        # Define Node 3: Format output
        def format_output(state: LatexConverterState) -> Dict:
            """
            Final formatting and metadata addition.
            
            Input:
            ------
            state["latex_code"]: str
                Raw LaTeX code from Gemini
            state["metadata"]["count"]: int
                Number of questions
            
            Processing:
            ----------
            - Add metadata comments
            - Final validation
            - Ensure proper formatting
            
            Output:
            ------
            Returns updates to state:
            {
                "latex_code": "% Metadata\\n\\n\\documentclass..."
            }
            
            Args:
                state: Current state
                
            Returns:
                Dict with state updates
            """
            latex = state["latex_code"]
            count = state["metadata"]["count"]
            
            print(f"[Node: format_output] Adding metadata and final formatting...")
            
            # Add metadata comments at the top
            formatted = f"""% Generated by LangGraph A2A Agent
% Questions Count: {count}
% Generator: Gemini 2.0 Flash
% Timestamp: {import_datetime_now()}

{latex}"""
            
            print(f"[Node: format_output] Formatting complete")
            
            # Return state updates
            return {"latex_code": formatted}
        
        # Add nodes to the graph
        workflow.add_node("parse_questions", parse_questions)
        workflow.add_node("convert_to_latex", convert_to_latex)
        workflow.add_node("format_output", format_output)
        
        # Define the edges (workflow flow)
        # Set entry point (where to start)
        workflow.set_entry_point("parse_questions")
        
        # Define transitions
        workflow.add_edge("parse_questions", "convert_to_latex")
        workflow.add_edge("convert_to_latex", "format_output")
        workflow.add_edge("format_output", END)
        
        # Compile the graph
        # This creates an executable workflow
        compiled_graph = workflow.compile()
        
        return compiled_graph
    
    async def convert_to_latex(self, questions: str) -> str:
        """
        Convert questions to LaTeX code using the state machine.
        
        This is the main public method that executes the workflow.
        
        Execution Flow:
        --------------
        1. Create initial state with questions
        2. Invoke graph (runs all nodes)
        3. Extract final LaTeX code from end state
        4. Return result
        
        Args:
            questions: Text containing questions to convert
            
        Returns:
            str: Complete LaTeX document code
            
        Example:
            >>> agent = LatexConverterAgent(api_key)
            >>> latex = await agent.convert_to_latex("1. What is AI?\\n2. How does ML work?")
            >>> print(latex)
            % Generated by LangGraph...
            \\documentclass{article}
            ...
        """
        print(f"[LaTeX Converter] Starting conversion workflow")
        
        # Step 1: Initialize state
        initial_state: LatexConverterState = {
            "questions": questions,
            "latex_code": "",
            "metadata": {}
        }
        
        # Step 2: Invoke the graph
        # This executes the state machine:
        # - Starts at entry point (parse_questions)
        # - Follows edges through all nodes
        # - Accumulates state updates
        # - Stops at END
        print(f"[LaTeX Converter] Executing state graph...")
        final_state = self.graph.invoke(initial_state)
        
        # Step 3: Extract final result
        latex_code = final_state["latex_code"]
        
        print(f"[LaTeX Converter] Workflow complete")
        print(f"[LaTeX Converter] Generated LaTeX: {len(latex_code)} characters")
        
        return latex_code


def import_datetime_now() -> str:
    """Helper function to get current timestamp."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Example usage (for testing)
if __name__ == "__main__":
    import asyncio
    
    # Load API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        exit(1)
    
    # Create agent and convert
    agent = LatexConverterAgent(api_key)
    
    async def test():
        questions = """1. What is machine learning?
2. How do neural networks learn?
3. What is the difference between AI and ML?
4. Explain gradient descent
5. What are activation functions?"""
        
        latex = await agent.convert_to_latex(questions)
        
        print("\n" + "="*80)
        print("GENERATED LATEX CODE:")
        print("="*80)
        print(latex)
        
        # Optionally save to file
        with open("output.tex", "w") as f:
            f.write(latex)
        print("\nSaved to output.tex")
    
    asyncio.run(test())