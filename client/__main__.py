"""
Client Agent - Entry Point

This is the client that orchestrates the multi-agent workflow using
Google ADK and A2A protocol.

Usage:
    python -m client "Machine Learning"
    python -m client "Quantum Computing" --verbose
"""

import os
import sys
import argparse
import asyncio
from dotenv import load_dotenv

from .client_agent import run_workflow


def main():
    """
    Main entry point for the client agent.
    
    Flow:
    1. Load environment variables
    2. Parse command-line arguments
    3. Run the workflow
    4. Display results
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Please set it in .env file or export it:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run Multi-Agent A2A Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m client "Machine Learning"
  python -m client "Quantum Computing" --verbose
  python -m client "Ancient Rome" --save output.tex

The workflow:
  1. Client calls CrewAI agent (port 8001) to generate questions
  2. Client calls LangGraph agent (port 8002) to convert to LaTeX
  3. Client displays/saves the final LaTeX code
        """
    )
    
    parser.add_argument(
        "topic",
        type=str,
        help="Topic to generate questions about"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--save",
        "-s",
        type=str,
        metavar="FILE",
        help="Save LaTeX output to file"
    )
    
    parser.add_argument(
        "--crewai-url",
        type=str,
        default="http://localhost:8001",
        help="CrewAI agent URL (default: http://localhost:8001)"
    )
    
    parser.add_argument(
        "--langgraph-url",
        type=str,
        default="http://localhost:8002",
        help="LangGraph agent URL (default: http://localhost:8002)"
    )
    
    args = parser.parse_args()
    
    # Display header
    print("\n" + "="*80)
    print("MULTI-AGENT A2A WORKFLOW CLIENT")
    print("="*80)
    print(f"Topic: {args.topic}")
    print(f"CrewAI Agent: {args.crewai_url}")
    print(f"LangGraph Agent: {args.langgraph_url}")
    print("="*80 + "\n")
    
    # Run workflow
    try:
        latex_code = asyncio.run(
            run_workflow(
                topic=args.topic,
                api_key=api_key,
                crewai_url=args.crewai_url,
                langgraph_url=args.langgraph_url,
                verbose=args.verbose
            )
        )
        
        # Display results
        print("\n" + "="*80)
        print("WORKFLOW COMPLETE - FINAL LATEX CODE")
        print("="*80)
        print(latex_code)
        print("="*80 + "\n")
        
        # Save to file if requested
        if args.save:
            with open(args.save, 'w') as f:
                f.write(latex_code)
            print(f"✓ Saved to {args.save}\n")
        
        print("✓ Success! You can now compile the LaTeX code.")
        
    except KeyboardInterrupt:
        print("\n\n[Client] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[Client] ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()