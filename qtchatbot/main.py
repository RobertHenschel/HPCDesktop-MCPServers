#!/usr/bin/env python3
"""
HPC MCP Chatbot - Main Entry Point

A Qt-based chatbot with MCP tool integration for HPC cluster management.
Uses the RealLMS API for LLM capabilities and loads MCP servers for tool access.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

from chatbot import ChatbotWindow
from llm_client import LLMClient
from mcp_manager import MCPManager


def load_config(config_path: str) -> dict:
    """Load configuration from config.dat file."""
    config = {}
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    
    return config


def load_system_prompt(prompts_dir: str, mcp_manager: MCPManager) -> str:
    """Load and format the system prompt."""
    prompt_path = os.path.join(prompts_dir, 'system_prompt.txt')
    
    if not os.path.exists(prompt_path):
        # Default prompt if file doesn't exist
        return "You are an HPC assistant with access to cluster management tools."
    
    with open(prompt_path, 'r') as f:
        template = f.read()
    
    # Get tool descriptions and substitute
    tool_descriptions = mcp_manager.get_tool_descriptions()
    
    return template.format(tool_descriptions=tool_descriptions)


def main():
    """Main entry point."""
    print("=" * 60)
    print("HPC MCP Chatbot")
    print("=" * 60)
    
    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Parent directory
    
    config_path = os.path.join(base_dir, 'config.dat')
    mcps_config_path = os.path.join(base_dir, 'available_mcps.json')
    prompts_dir = os.path.join(script_dir, 'prompts')
    
    # Load configuration
    print(f"Loading config from: {config_path}")
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    api_key = config.get('REALLMSAPI')
    hostname = config.get('hostname')
    model = config.get('model')
    
    if not api_key or not hostname:
        print("ERROR: config.dat must contain REALLMSAPI and hostname")
        sys.exit(1)
    
    if not model:
        model = config.get('modelbackup', 'llama-4-scout')
        print(f"WARNING: No model specified, using: {model}")
    
    print(f"API Host: {hostname}")
    print(f"Model: {model}")
    print("-" * 60)
    
    # Initialize LLM client
    llm_client = LLMClient(api_key, hostname, model)
    
    # Check connection
    connected, msg = llm_client.check_connection()
    if connected:
        print(f"✓ Connected to RealLMS API")
    else:
        print(f"⚠ Connection check failed: {msg}")
        print("  (Will retry when chatbot starts)")
    
    # Initialize MCP Manager
    print(f"\nLoading MCPs from: {mcps_config_path}")
    mcp_manager = MCPManager(base_dir, mcps_config_path)
    mcp_manager.load_mcps()
    
    # List loaded tools
    tools = mcp_manager.list_tools()
    if tools:
        print(f"✓ Loaded {len(tools)} tools: {', '.join(tools)}")
    else:
        print("⚠ No tools loaded")
    
    # Load system prompt
    print(f"\nLoading prompts from: {prompts_dir}")
    system_prompt = load_system_prompt(prompts_dir, mcp_manager)
    print(f"✓ System prompt loaded ({len(system_prompt)} chars)")
    
    print("-" * 60)
    print("Starting chatbot...")
    print("=" * 60)
    
    # Create Qt application FIRST before any Qt objects
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("HPC MCP Chatbot")
    app.setApplicationDisplayName("HPC MCP Chatbot")
    app.setOrganizationName("Indiana University")
    
    # Use a common font that's likely to exist
    font = QFont()
    font.setFamily("Monospace")
    font.setPointSize(11)
    app.setFont(font)
    
    # Create main window
    try:
        window = ChatbotWindow(llm_client, mcp_manager, system_prompt)
        window.show()
        
        # Run the event loop
        return_code = app.exec()
        sys.exit(return_code)
        
    except Exception as e:
        print(f"ERROR: Failed to start chatbot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
