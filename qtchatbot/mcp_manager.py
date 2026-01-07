"""MCP Manager for loading and executing MCP server tools."""

import json
import os
import sys
import importlib.util
import inspect
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    server_name: str


@dataclass
class MCPServer:
    """Represents an MCP server."""
    name: str
    description: str
    path: str
    tools: List[MCPTool] = field(default_factory=list)


class MCPManager:
    """Manages MCP servers and their tools."""
    
    def __init__(self, base_path: str, mcps_config_path: str):
        """
        Initialize the MCP Manager.
        
        Args:
            base_path: Base directory for resolving MCP paths
            mcps_config_path: Path to available_mcps.json
        """
        self.base_path = base_path
        self.mcps_config_path = mcps_config_path
        self.servers: Dict[str, MCPServer] = {}
        self.tools: Dict[str, MCPTool] = {}
    
    def load_mcps(self) -> None:
        """Load all MCP servers from the config file."""
        try:
            with open(self.mcps_config_path, 'r') as f:
                config = json.load(f)
            
            for mcp_config in config.get('mcps', []):
                self._load_mcp_server(mcp_config)
                
        except FileNotFoundError:
            print(f"Warning: MCP config file not found: {self.mcps_config_path}")
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in MCP config: {e}")
        except Exception as e:
            print(f"Warning: Error loading MCPs: {e}")
    
    def _load_mcp_server(self, mcp_config: Dict) -> None:
        """Load a single MCP server."""
        name = mcp_config.get('name', 'Unknown')
        description = mcp_config.get('description', '')
        rel_path = mcp_config.get('path', '')
        
        if not rel_path:
            print(f"Warning: MCP {name} has no path specified")
            return
        
        # Resolve the full path
        full_path = os.path.join(self.base_path, rel_path)
        
        if not os.path.exists(full_path):
            print(f"Warning: MCP server file not found: {full_path}")
            return
        
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(f"mcp_{name.lower()}", full_path)
            module = importlib.util.module_from_spec(spec)
            
            # Add the module's directory to sys.path temporarily
            module_dir = os.path.dirname(full_path)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            
            spec.loader.exec_module(module)
            
            # Create server object
            server = MCPServer(
                name=name,
                description=description,
                path=full_path
            )
            
            # Find the FastMCP instance and extract tools
            mcp_instance = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, '_tool_manager'):
                    mcp_instance = attr
                    break
            
            if mcp_instance:
                self._extract_tools(server, mcp_instance, module)
            else:
                # Fall back to finding decorated functions directly
                self._extract_tools_from_module(server, module)
            
            self.servers[name] = server
            print(f"Loaded MCP server '{name}' with {len(server.tools)} tools")
            
        except Exception as e:
            print(f"Warning: Error loading MCP server {name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_tools(self, server: MCPServer, mcp_instance: Any, module: Any) -> None:
        """Extract tools from a FastMCP instance."""
        # Try to get tools from the tool manager
        if hasattr(mcp_instance, '_tool_manager'):
            tool_manager = mcp_instance._tool_manager
            if hasattr(tool_manager, 'tools'):
                for tool_name, tool_info in tool_manager.tools.items():
                    self._register_tool(server, tool_name, tool_info, module)
                return
        
        # Fall back to module inspection
        self._extract_tools_from_module(server, module)
    
    def _extract_tools_from_module(self, server: MCPServer, module: Any) -> None:
        """Extract tools by inspecting module functions."""
        for name, func in inspect.getmembers(module, inspect.isfunction):
            # Skip private functions and the main entry point
            if name.startswith('_') or name == 'main':
                continue
            
            # Check if this looks like an MCP tool (has docstring)
            if func.__doc__:
                self._register_function_as_tool(server, name, func)
    
    def _register_tool(self, server: MCPServer, tool_name: str, tool_info: Any, module: Any) -> None:
        """Register a tool from tool manager info."""
        # Get the actual function
        func = getattr(module, tool_name, None)
        if not func:
            return
        
        # Extract description from docstring or tool info
        description = ""
        if hasattr(tool_info, 'description'):
            description = tool_info.description
        elif func.__doc__:
            description = func.__doc__.strip().split('\n')[0]
        
        # Extract parameters from function signature
        parameters = self._get_function_parameters(func)
        
        tool = MCPTool(
            name=tool_name,
            description=description,
            parameters=parameters,
            function=func,
            server_name=server.name
        )
        
        server.tools.append(tool)
        self.tools[tool_name] = tool
    
    def _register_function_as_tool(self, server: MCPServer, name: str, func: Callable) -> None:
        """Register a function as a tool."""
        # Get description from docstring
        description = ""
        if func.__doc__:
            # Get first line of docstring
            description = func.__doc__.strip().split('\n')[0]
        
        # Extract parameters
        parameters = self._get_function_parameters(func)
        
        tool = MCPTool(
            name=name,
            description=description,
            parameters=parameters,
            function=func,
            server_name=server.name
        )
        
        server.tools.append(tool)
        self.tools[name] = tool
    
    def _get_function_parameters(self, func: Callable) -> Dict[str, Any]:
        """Get OpenAI-compatible parameter schema from function signature."""
        sig = inspect.signature(func)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
            
            param_type = "string"  # Default type
            
            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list or param.annotation == List:
                    param_type = "array"
            
            properties[param_name] = {"type": param_type}
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def get_tools_for_llm(self) -> List[Dict]:
        """Get tool definitions in OpenAI function calling format."""
        tools = []
        
        for tool in self.tools.values():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        
        return tools
    
    def get_tool_descriptions(self) -> str:
        """Get human-readable tool descriptions."""
        descriptions = []
        
        for server in self.servers.values():
            descriptions.append(f"\n## {server.name}")
            if server.description:
                descriptions.append(f"{server.description}")
            
            for tool in server.tools:
                descriptions.append(f"\n- **{tool.name}**: {tool.description}")
        
        return "\n".join(descriptions)
    
    def execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """
        Execute a tool by name with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool result as a string
        """
        if tool_name not in self.tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        tool = self.tools[tool_name]
        
        try:
            result = tool.function(**arguments)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as e:
            return json.dumps({"error": f"Tool execution error: {str(e)}"})
    
    def list_servers(self) -> List[str]:
        """List all loaded server names."""
        return list(self.servers.keys())
    
    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self.tools.keys())

