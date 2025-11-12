#!/usr/bin/env python3
"""
Run All Script - Start all agents and run workflow

This script:
1. Starts CrewAI agent in background
2. Starts LangGraph agent in background
3. Waits for agents to initialize
4. Runs the client workflow
5. Cleans up background processes

Usage:
    python run_all.py "Machine Learning"
    python run_all.py "Quantum Computing" --verbose
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import asyncio
from dotenv import load_dotenv

# Import client workflow
from client.client_agent import run_workflow


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def start_agent_process(module_name: str, port: int):
    """
    Start an agent as a background process.
    
    Args:
        module_name: Python module name (e.g., "crewai_question_agent")
        port: Port number for the agent
        
    Returns:
        subprocess.Popen: Process object
    """
    print(f"[Run All] Starting {module_name} on port {port}...")
    
    # Start process in background
    process = subprocess.Popen(
        [sys.executable, "-m", module_name, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process


def wait_for_agents(timeout: int = 30):
    """
    Wait for agents to start and be ready.
    
    Args:
        timeout: Maximum time to wait in seconds
    """
    import httpx
    
    agents = {
        "CrewAI": "http://localhost:8001/.well-known/agent.json",
        "LangGraph": "http://localhost:8002/.well-known/agent.json"
    }
    
    print(f"[Run All] Waiting for agents to initialize (timeout: {timeout}s)...")
    
    start_time = time.time()
    
    for agent_name, url in agents.items():
        ready = False
        
        while not ready and (time.time() - start_time) < timeout:
            try:
                response = httpx.get(url, timeout=2.0)
                if response.status_code == 200:
                    print(f"[Run All] ✓ {agent_name} agent ready")
                    ready = True
                else:
                    time.sleep(1)
            except Exception:
                time.sleep(1)
        
        if not ready:
            raise TimeoutError(f"{agent_name} agent did not start within {timeout}s")
    
    print(f"[Run All] ✓ All agents initialized\n")


def cleanup_processes(processes):
    """
    Terminate background processes.
    
    Args:
        processes: List of subprocess.Popen objects
    """
    print("\n[Run All] Cleaning up background processes...")
    
    for process in processes:
        try:
            process.terminate()
            process.wait(timeout=5)
            print(f"[Run All] ✓ Process {process.pid} terminated")
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"[Run All] ✗ Process {process.pid} killed (force)")
        except Exception as e:
            print(f"[Run All] Warning: {e}")


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Please set it in .env file or export it:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run complete multi-agent A2A workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py "Machine Learning"
  python run_all.py "Quantum Computing" --verbose
  python run_all.py "Ancient Rome" --save output.tex

This script will:
  1. Start CrewAI agent (port 8001)
  2. Start LangGraph agent (port 8002)
  3. Run client workflow
  4. Clean up processes
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
        "--timeout",
        "-t",
        type=int,
        default=30,
        help="Agent startup timeout in seconds (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Display header
    print_header("MULTI-AGENT A2A SYSTEM - RUN ALL")
    print(f"Topic: {args.topic}")
    print(f"Verbose: {args.verbose}")
    if args.save:
        print(f"Output file: {args.save}")
    print()
    
    processes = []
    
    try:
        # =================================================================
        # STEP 1: Start remote agents
        # =================================================================
        print_header("STEP 1: Starting Remote Agents")
        
        # Start CrewAI agent
        crewai_process = start_agent_process("crewai_question_agent", 8001)
        processes.append(crewai_process)
        
        # Start LangGraph agent
        langgraph_process = start_agent_process("langgraph_latex_agent", 8002)
        processes.append(langgraph_process)
        
        # Wait for agents to be ready
        wait_for_agents(timeout=args.timeout)
        
        # =================================================================
        # STEP 2: Run client workflow
        # =================================================================
        print_header("STEP 2: Running Client Workflow")
        
        latex_code = asyncio.run(
            run_workflow(
                topic=args.topic,
                api_key=api_key,
                verbose=args.verbose
            )
        )
        
        # =================================================================
        # STEP 3: Display results
        # =================================================================
        print_header("STEP 3: Results")
        
        print("FINAL LATEX CODE:")
        print("="*80)
        print(latex_code)
        print("="*80 + "\n")
        
        # Save to file if requested
        if args.save:
            with open(args.save, 'w') as f:
                f.write(latex_code)
            print(f"✓ Saved to {args.save}\n")
        
        print("✓ Workflow completed successfully!")
        print("\nYou can now compile the LaTeX code with:")
        if args.save:
            print(f"  pdflatex {args.save}")
        else:
            print("  pdflatex <your-file>.tex")
        
    except KeyboardInterrupt:
        print("\n\n[Run All] Interrupted by user")
        cleanup_processes(processes)
        sys.exit(130)
    
    except Exception as e:
        print(f"\n\n[Run All] ERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        cleanup_processes(processes)
        sys.exit(1)
    
    finally:
        # Always cleanup
        if processes:
            cleanup_processes(processes)
    
    print("\n[Run All] Done!")


if __name__ == "__main__":
    main()