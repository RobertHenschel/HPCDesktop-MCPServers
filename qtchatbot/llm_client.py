"""LLM Client for RealLMS API with prompt-based tool calling support."""

import json
import re
import requests
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a tool call parsed from the LLM response."""
    name: str
    arguments: Dict


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: str
    tool_calls: List[ToolCall]


class LLMClient:
    """Client for RealLMS API with prompt-based tool calling support."""
    
    # Pattern to match tool call blocks wrapped in backticks
    TOOL_CALL_BACKTICK_PATTERN = re.compile(
        r'```(?:tool_calls?|json)?\s*\n?\s*(\{[^`]*?"tool"\s*:\s*"[^"]+?"[^`]*?\})\s*\n?```',
        re.DOTALL
    )
    
    # Pattern to match raw JSON tool calls (without backticks)
    # Matches {"tool": "...", "arguments": {...}} possibly spanning multiple lines
    TOOL_CALL_RAW_PATTERN = re.compile(
        r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\}|\[\])\s*\}',
        re.DOTALL
    )
    
    # Alternative pattern for when arguments comes first
    TOOL_CALL_RAW_ALT_PATTERN = re.compile(
        r'\{\s*"arguments"\s*:\s*(\{[^}]*\}|\[\])\s*,\s*"tool"\s*:\s*"([^"]+)"\s*\}',
        re.DOTALL
    )
    
    def __init__(self, api_key: str, hostname: str, model: str):
        self.api_key = api_key
        self.model = model
        
        # Normalize hostname
        if not hostname.startswith('http://') and not hostname.startswith('https://'):
            hostname = f'https://{hostname}'
        self.base_url = hostname
        self.endpoint = f"{self.base_url}/direct/v1/chat/completions"
    
    def _get_headers(self) -> Dict:
        """Get request headers."""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def parse_tool_calls(self, content: str) -> List[ToolCall]:
        """
        Parse tool call blocks from response content.
        
        Handles multiple formats:
        1. Wrapped in backticks: ```tool_call {"tool": "...", "arguments": {}} ```
        2. Raw JSON: {"tool": "...", "arguments": {}}
        3. Alternative order: {"arguments": {}, "tool": "..."}
        """
        tool_calls = []
        found_tools = set()  # Avoid duplicates
        
        # Method 1: Try backtick-wrapped format first (most explicit)
        for match in self.TOOL_CALL_BACKTICK_PATTERN.finditer(content):
            try:
                json_str = match.group(1)
                data = json.loads(json_str)
                
                tool_name = data.get('tool')
                arguments = data.get('arguments', {})
                
                if tool_name and tool_name not in found_tools:
                    tool_calls.append(ToolCall(
                        name=tool_name,
                        arguments=arguments if isinstance(arguments, dict) else {}
                    ))
                    found_tools.add(tool_name)
            except json.JSONDecodeError:
                continue
        
        # Method 2: Try raw JSON pattern {"tool": "...", "arguments": {...}}
        for match in self.TOOL_CALL_RAW_PATTERN.finditer(content):
            try:
                tool_name = match.group(1)
                args_str = match.group(2)
                arguments = json.loads(args_str)
                
                if tool_name and tool_name not in found_tools:
                    tool_calls.append(ToolCall(
                        name=tool_name,
                        arguments=arguments if isinstance(arguments, dict) else {}
                    ))
                    found_tools.add(tool_name)
            except json.JSONDecodeError:
                continue
        
        # Method 3: Try alternative order {"arguments": {...}, "tool": "..."}
        for match in self.TOOL_CALL_RAW_ALT_PATTERN.finditer(content):
            try:
                args_str = match.group(1)
                tool_name = match.group(2)
                arguments = json.loads(args_str)
                
                if tool_name and tool_name not in found_tools:
                    tool_calls.append(ToolCall(
                        name=tool_name,
                        arguments=arguments if isinstance(arguments, dict) else {}
                    ))
                    found_tools.add(tool_name)
            except json.JSONDecodeError:
                continue
        
        # Method 4: Fallback - try to find any JSON object with "tool" key
        if not tool_calls:
            # Look for any JSON-like structure with "tool" in it
            json_pattern = re.compile(r'\{[^{}]*"tool"[^{}]*\}')
            for match in json_pattern.finditer(content):
                try:
                    data = json.loads(match.group(0))
                    tool_name = data.get('tool')
                    arguments = data.get('arguments', {})
                    
                    if tool_name and tool_name not in found_tools:
                        tool_calls.append(ToolCall(
                            name=tool_name,
                            arguments=arguments if isinstance(arguments, dict) else {}
                        ))
                        found_tools.add(tool_name)
                except json.JSONDecodeError:
                    continue
        
        return tool_calls
    
    def chat(
        self,
        messages: List[Dict],
        timeout: int = 300
    ) -> LLMResponse:
        """
        Send a chat request to the LLM (non-streaming).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            timeout: Request timeout in seconds
            
        Returns:
            LLMResponse with content and any parsed tool calls
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        
        response = requests.post(
            self.endpoint,
            headers=self._get_headers(),
            json=payload,
            timeout=timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text}")
        
        data = response.json()
        content = data['choices'][0]['message'].get('content', '') or ''
        
        # Parse tool calls from content
        tool_calls = self.parse_tool_calls(content)
        
        return LLMResponse(content=content, tool_calls=tool_calls)
    
    def chat_stream(
        self,
        messages: List[Dict],
        on_chunk: Callable[[str], None],
        timeout: int = 300
    ) -> LLMResponse:
        """
        Stream a chat response from the LLM.
        
        Args:
            messages: List of message dicts
            on_chunk: Callback function called with each content chunk
            timeout: Request timeout
            
        Returns:
            LLMResponse with complete content and tool calls
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }
        
        response = requests.post(
            self.endpoint,
            headers=self._get_headers(),
            json=payload,
            stream=True,
            timeout=timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text}")
        
        full_content = ""
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            
            # Skip SSE prefix
            if line_str.startswith('data: '):
                line_str = line_str[6:]
            
            # Skip [DONE] marker
            if line_str.strip() == '[DONE]':
                break
            
            try:
                chunk = json.loads(line_str)
                
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        full_content += content
                        on_chunk(content)
                        
            except json.JSONDecodeError:
                continue
        
        # Parse tool calls from complete content
        tool_calls = self.parse_tool_calls(full_content)
        
        return LLMResponse(content=full_content, tool_calls=tool_calls)
    
    def check_connection(self) -> tuple[bool, str]:
        """Check connection to the API."""
        try:
            response = requests.get(
                f"{self.base_url}/direct/v1/models",
                headers=self._get_headers(),
                timeout=5
            )
            if response.status_code == 200:
                return True, "Connected"
            else:
                return False, f"Error: {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Connection failed"
        except requests.exceptions.Timeout:
            return False, "Timeout"
        except Exception as e:
            return False, f"Error: {str(e)[:30]}"
