"""
Shared Utilities

Common utility functions used across the multi-agent system.
"""

import os
import sys
from datetime import datetime
from typing import Optional


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        str: Timestamp like "2025-11-12 14:30:00"
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_header(title: str, width: int = 80, char: str = "="):
    """
    Print a formatted header.
    
    Args:
        title: Header title
        width: Total width of header
        char: Character to use for border
    """
    print("\n" + char * width)
    print(title.center(width))
    print(char * width + "\n")


def print_section(title: str, width: int = 80):
    """
    Print a section divider.
    
    Args:
        title: Section title
        width: Total width
    """
    print(f"\n{title}")
    print("-" * width)


def validate_env_var(var_name: str, required: bool = True) -> Optional[str]:
    """
    Validate and get environment variable.
    
    Args:
        var_name: Name of environment variable
        required: Whether the variable is required
        
    Returns:
        str or None: Environment variable value
        
    Raises:
        SystemExit: If required variable is not set
    """
    value = os.getenv(var_name)
    
    if required and not value:
        print(f"ERROR: {var_name} environment variable not set")
        print(f"Please set it in .env file or export it:")
        print(f"  export {var_name}='your-value-here'")
        sys.exit(1)
    
    return value


def check_port_available(host: str, port: int) -> bool:
    """
    Check if a port is available.
    
    Args:
        host: Host to check
        port: Port to check
        
    Returns:
        bool: True if port is available
    """
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect((host, port))
            return False  # Port is in use
    except (socket.timeout, ConnectionRefusedError):
        return True  # Port is available
    except Exception as e:
        print(f"Warning: Could not check port {port}: {e}")
        return True


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size (e.g., "1.5 KB", "2.3 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


class ColoredOutput:
    """Helper class for colored console output."""
    
    # ANSI color codes
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    
    @staticmethod
    def success(message: str) -> str:
        """Format success message in green."""
        return f"{ColoredOutput.GREEN}✓ {message}{ColoredOutput.RESET}"
    
    @staticmethod
    def error(message: str) -> str:
        """Format error message in red."""
        return f"{ColoredOutput.RED}✗ {message}{ColoredOutput.RESET}"
    
    @staticmethod
    def warning(message: str) -> str:
        """Format warning message in yellow."""
        return f"{ColoredOutput.YELLOW}⚠ {message}{ColoredOutput.RESET}"
    
    @staticmethod
    def info(message: str) -> str:
        """Format info message in blue."""
        return f"{ColoredOutput.BLUE}ℹ {message}{ColoredOutput.RESET}"
    
    @staticmethod
    def bold(message: str) -> str:
        """Format message in bold."""
        return f"{ColoredOutput.BOLD}{message}{ColoredOutput.RESET}"


# Example usage
if __name__ == "__main__":
    print_header("Shared Utilities Test")
    
    print(ColoredOutput.success("This is a success message"))
    print(ColoredOutput.error("This is an error message"))
    print(ColoredOutput.warning("This is a warning message"))
    print(ColoredOutput.info("This is an info message"))
    
    print(f"\nTimestamp: {get_timestamp()}")
    print(f"Truncated: {truncate_text('This is a very long text' * 10, 50)}")
    print(f"File size: {format_file_size(1234567)}")