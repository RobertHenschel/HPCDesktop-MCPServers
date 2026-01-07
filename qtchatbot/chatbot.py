"""Qt Chatbot Application with MCP Tool Support (Prompt-based)."""

import os
import json
from typing import List, Dict, Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QFrame, QScrollArea, QSizePolicy,
    QStatusBar, QSplitter, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex
from PyQt6.QtGui import QFont, QKeyEvent, QIcon, QPixmap

from llm_client import LLMClient, LLMResponse, ToolCall
from mcp_manager import MCPManager


class ChatWorker(QThread):
    """Worker thread for handling LLM requests with prompt-based tool execution."""
    
    chunk_received = pyqtSignal(str)  # Emits content chunks
    tool_call_detected = pyqtSignal(str)  # Emits tool name being called
    tool_result_received = pyqtSignal(str, str)  # Emits (tool_name, result)
    finished = pyqtSignal(str)  # Emits complete response
    error_occurred = pyqtSignal(str)  # Emits error messages
    
    MAX_TOOL_ITERATIONS = 5  # Prevent infinite loops
    
    def __init__(
        self,
        llm_client: LLMClient,
        mcp_manager: MCPManager,
        messages: List[Dict]
    ):
        super().__init__()
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager
        self.messages = messages.copy()
        self.is_cancelled = False
        self.mutex = QMutex()
    
    def _emit_chunk(self, chunk: str):
        """Thread-safe chunk emission."""
        self.mutex.lock()
        cancelled = self.is_cancelled
        self.mutex.unlock()
        
        if not cancelled:
            self.chunk_received.emit(chunk)
    
    def run(self):
        """Execute the chat request with tool handling."""
        try:
            iterations = 0
            final_content = ""
            
            while iterations < self.MAX_TOOL_ITERATIONS:
                self.mutex.lock()
                cancelled = self.is_cancelled
                self.mutex.unlock()
                
                if cancelled:
                    break
                    
                iterations += 1
                
                # Stream the response with callback
                try:
                    response = self.llm_client.chat_stream(
                        self.messages,
                        on_chunk=self._emit_chunk
                    )
                except Exception as e:
                    # If streaming fails, try non-streaming
                    response = self.llm_client.chat(self.messages)
                    if response.content:
                        self._emit_chunk(response.content)
                
                self.mutex.lock()
                cancelled = self.is_cancelled
                self.mutex.unlock()
                
                if cancelled:
                    break
                
                final_content = response.content
                
                # Check for tool calls
                if response.tool_calls:
                    # Add assistant message to history
                    self.messages.append({
                        "role": "assistant",
                        "content": response.content
                    })
                    
                    # Execute each tool call
                    tool_results = []
                    for tool_call in response.tool_calls:
                        self.mutex.lock()
                        cancelled = self.is_cancelled
                        self.mutex.unlock()
                        
                        if cancelled:
                            break
                        
                        self.tool_call_detected.emit(tool_call.name)
                        
                        # Execute the tool
                        result = self.mcp_manager.execute_tool(
                            tool_call.name,
                            tool_call.arguments
                        )
                        
                        self.tool_result_received.emit(tool_call.name, result)
                        tool_results.append(f"Tool: {tool_call.name}\nResult:\n{result}")
                    
                    self.mutex.lock()
                    cancelled = self.is_cancelled
                    self.mutex.unlock()
                    
                    if tool_results and not cancelled:
                        # Add tool results as a user message to continue the conversation
                        results_message = "Here are the tool results:\n\n" + "\n\n".join(tool_results)
                        results_message += "\n\nPlease interpret these results for the user."
                        
                        self.messages.append({
                            "role": "user", 
                            "content": results_message
                        })
                        
                        # Continue loop to get LLM interpretation
                        continue
                else:
                    # No tool calls, we're done
                    break
            
            self.mutex.lock()
            cancelled = self.is_cancelled
            self.mutex.unlock()
            
            if not cancelled:
                self.finished.emit(final_content)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))
    
    def cancel(self):
        """Cancel the request."""
        self.mutex.lock()
        self.is_cancelled = True
        self.mutex.unlock()


class MessageWidget(QFrame):
    """Widget for displaying individual chat messages."""
    
    def __init__(self, content: str, role: str = "user", parent=None):
        super().__init__(parent)
        self.role = role
        self.full_content = content
        
        # Style based on role
        if role == "user":
            self.setStyleSheet("""
                MessageWidget {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #6366f1, stop:1 #4f46e5);
                    border-radius: 12px;
                    padding: 12px;
                    margin: 4px;
                }
            """)
            text_color = "white"
        elif role == "tool":
            self.setStyleSheet("""
                MessageWidget {
                    background: #fef3c7;
                    border: 2px solid #f59e0b;
                    border-radius: 12px;
                    padding: 12px;
                    margin: 4px;
                }
            """)
            text_color = "#92400e"
        else:
            self.setStyleSheet("""
                MessageWidget {
                    background: white;
                    border: 2px solid #e5e7eb;
                    border-radius: 12px;
                    padding: 12px;
                    margin: 4px;
                }
            """)
            text_color = "#1e293b"
        
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        
        self.content_label = QLabel(content if content else "...")
        self.content_label.setWordWrap(True)
        self.content_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.content_label.setStyleSheet(f"color: {text_color}; background: transparent; border: none;")
        
        font = QFont("Monospace", 11)
        self.content_label.setFont(font)
        
        layout.addWidget(self.content_label)
        self.setLayout(layout)
        
        # Size constraints
        self.setMinimumWidth(200)
        self.setMaximumWidth(700)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    
    def update_content(self, content: str):
        """Update message content."""
        self.full_content = content
        self.content_label.setText(content if content else "...")
    
    def append_content(self, chunk: str):
        """Append content chunk."""
        self.full_content += chunk
        self.content_label.setText(self.full_content)


class ChatbotWindow(QMainWindow):
    """Main chatbot window."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        mcp_manager: MCPManager,
        system_prompt: str
    ):
        super().__init__()
        
        self.llm_client = llm_client
        self.mcp_manager = mcp_manager
        self.system_prompt = system_prompt
        
        self.conversation_history: List[Dict] = []
        self.current_worker: Optional[ChatWorker] = None
        self.current_message_widget: Optional[MessageWidget] = None
        
        # Track tool message containers for show/hide functionality
        self.tool_message_containers: List[QWidget] = []
        self.show_tool_messages = True
        
        self.init_ui()
        self.init_status_bar()
        
        # Connection check timer (start after window is fully shown)
        self.connection_timer = QTimer(self)
        self.connection_timer.timeout.connect(self._safe_update_connection)
        
        # Delay starting the timer until after the window is shown
        QTimer.singleShot(1000, self._start_connection_timer)
    
    def _start_connection_timer(self):
        """Start the connection timer after window is ready."""
        if not self.isVisible():
            return
        self._safe_update_connection()
        self.connection_timer.start(10000)
    
    def _safe_update_connection(self):
        """Safely update connection status."""
        try:
            if self.isVisible():
                self.update_connection_status()
        except Exception as e:
            print(f"Connection check error: {e}")
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("HPC MCP Chatbot")
        self.setGeometry(100, 100, 900, 700)
        
        # Try to set window icon
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Cray-1-AI.png')
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            if not pixmap.isNull():
                self.setWindowIcon(QIcon(pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio)))
        
        # Create splitter for resizable chat/input areas
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #334155;
                height: 4px;
            }
            QSplitter::handle:hover {
                background: #6366f1;
            }
        """)
        self.setCentralWidget(splitter)
        
        # Chat area
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet("""
            QScrollArea {
                background: #f8fafc;
                border: none;
            }
        """)
        
        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.chat_layout.setSpacing(8)
        self.chat_layout.setContentsMargins(16, 16, 16, 16)
        
        # Add stretch at start to push messages to bottom
        self.chat_layout.addStretch()
        
        self.chat_widget.setLayout(self.chat_layout)
        self.chat_scroll.setWidget(self.chat_widget)
        
        # Track insertion point
        self.message_insert_index = 1
        
        # Welcome message
        self._add_welcome_message()
        
        splitter.addWidget(self.chat_scroll)
        
        # Input area
        input_area = self._create_input_area()
        splitter.addWidget(input_area)
        
        # Set initial sizes (90% chat, 10% input)
        splitter.setSizes([600, 60])
        
        # Set minimum sizes
        self.chat_scroll.setMinimumHeight(100)
        input_area.setMinimumHeight(50)
        input_area.setMaximumHeight(150)
        
        # Main window style
        self.setStyleSheet("""
            QMainWindow {
                background: #f8fafc;
            }
        """)
    
    def _add_welcome_message(self):
        """Add welcome message."""
        tools_list = self.mcp_manager.list_tools()
        tools_str = ", ".join(tools_list) if tools_list else "No tools loaded"
        
        welcome_text = f"""Welcome to the HPC MCP Chatbot!

I have access to the following tools: {tools_str}

Ask me about your HPC cluster - I can check partitions, view your jobs, and more!"""
        
        msg = MessageWidget(welcome_text, role="assistant")
        container = self._wrap_message(msg, "assistant")
        self.chat_layout.insertWidget(self.message_insert_index, container)
        self.message_insert_index += 1
    
    def _wrap_message(self, widget: MessageWidget, role: str) -> QWidget:
        """Wrap a message widget with proper alignment."""
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        if role == "user":
            layout.addStretch()
            layout.addWidget(widget)
        else:
            layout.addWidget(widget)
            layout.addStretch()
        
        container.setLayout(layout)
        return container
    
    def _create_input_area(self) -> QWidget:
        """Create the input area."""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background: #1e293b;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        
        # Input field (single line height)
        self.input_field = QTextEdit()
        self.input_field.setPlaceholderText("Ask about your HPC cluster...")
        self.input_field.setFont(QFont("Monospace", 10))
        self.input_field.setFixedHeight(36)
        self.input_field.setStyleSheet("""
            QTextEdit {
                background: #0f172a;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 4px 8px;
            }
            QTextEdit:focus {
                border: 2px solid #6366f1;
            }
        """)
        self.input_field.keyPressEvent = self._input_key_press
        layout.addWidget(self.input_field, 1)
        
        # Show tool messages checkbox
        self.show_tools_checkbox = QCheckBox("Show system")
        self.show_tools_checkbox.setChecked(True)
        self.show_tools_checkbox.setFont(QFont("Monospace", 9))
        self.show_tools_checkbox.setStyleSheet("""
            QCheckBox {
                color: #94a3b8;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #475569;
                border-radius: 4px;
                background: #0f172a;
            }
            QCheckBox::indicator:checked {
                background: #f59e0b;
                border-color: #f59e0b;
            }
            QCheckBox::indicator:hover {
                border-color: #f59e0b;
            }
        """)
        self.show_tools_checkbox.stateChanged.connect(self._toggle_tool_messages)
        layout.addWidget(self.show_tools_checkbox)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.setFixedSize(70, 36)
        self.send_button.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
        self.send_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6366f1, stop:1 #4f46e5);
                color: white;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4f46e5, stop:1 #4338ca);
            }
            QPushButton:disabled {
                background: #475569;
                color: #94a3b8;
            }
        """)
        self.send_button.clicked.connect(self.send_message)
        layout.addWidget(self.send_button)
        
        # Stop button (hidden initially)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setFixedSize(70, 36)
        self.stop_button.setFont(QFont("Monospace", 10, QFont.Weight.Bold))
        self.stop_button.setStyleSheet("""
            QPushButton {
                background: #dc2626;
                color: white;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #b91c1c;
            }
        """)
        self.stop_button.clicked.connect(self.stop_request)
        self.stop_button.hide()
        layout.addWidget(self.stop_button)
        
        frame.setLayout(layout)
        return frame
    
    def _input_key_press(self, event: QKeyEvent):
        """Handle input key press."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                QTextEdit.keyPressEvent(self.input_field, event)
            else:
                self.send_message()
                event.accept()
                return
        else:
            QTextEdit.keyPressEvent(self.input_field, event)
    
    def _toggle_tool_messages(self, state: int):
        """Toggle visibility of tool/system messages."""
        self.show_tool_messages = (state == Qt.CheckState.Checked.value)
        
        # Update visibility of all tracked tool message containers
        for container in self.tool_message_containers:
            container.setVisible(self.show_tool_messages)
        
        # Scroll to maintain position
        QTimer.singleShot(50, self._scroll_to_bottom)
    
    def send_message(self):
        """Send a message."""
        message = self.input_field.toPlainText().strip()
        if not message:
            return
        
        # Clear input
        self.input_field.clear()
        
        # Add user message to UI
        user_msg = MessageWidget(message, role="user")
        container = self._wrap_message(user_msg, "user")
        self.chat_layout.insertWidget(self.message_insert_index, container)
        self.message_insert_index += 1
        
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Disable input
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.stop_button.show()
        
        # Create assistant message widget
        assistant_msg = MessageWidget("", role="assistant")
        container = self._wrap_message(assistant_msg, "assistant")
        self.chat_layout.insertWidget(self.message_insert_index, container)
        self.message_insert_index += 1
        self.current_message_widget = assistant_msg
        
        # Scroll to bottom
        self._scroll_to_bottom()
        
        # Build messages with system prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        
        # Clean up any previous worker first
        self._cleanup_worker()
        
        # Start worker (no tools parameter - using prompt-based tool calling)
        self.current_worker = ChatWorker(
            self.llm_client,
            self.mcp_manager,
            messages
        )
        self.current_worker.chunk_received.connect(self._on_chunk)
        self.current_worker.tool_call_detected.connect(self._on_tool_call_detected)
        self.current_worker.tool_result_received.connect(self._on_tool_result)
        self.current_worker.finished.connect(self._on_finished)
        self.current_worker.error_occurred.connect(self._on_error)
        # Connect to QThread's finished signal for cleanup
        self.current_worker.finished.connect(self._on_worker_thread_finished)
        self.current_worker.start()
    
    def _cleanup_worker(self):
        """Clean up the current worker thread properly."""
        if self.current_worker is not None:
            # Disconnect signals first
            try:
                self.current_worker.chunk_received.disconnect()
                self.current_worker.tool_call_detected.disconnect()
                self.current_worker.tool_result_received.disconnect()
                self.current_worker.finished.disconnect()
                self.current_worker.error_occurred.disconnect()
            except:
                pass
            
            # Cancel and wait for thread
            self.current_worker.cancel()
            if self.current_worker.isRunning():
                self.current_worker.wait(3000)  # Wait up to 3 seconds
            
            # Schedule for deletion
            self.current_worker.deleteLater()
            self.current_worker = None
    
    def _on_worker_thread_finished(self):
        """Called when the worker thread actually finishes."""
        # Don't set to None here - let _cleanup_worker handle it
        pass
    
    def _on_chunk(self, chunk: str):
        """Handle content chunk."""
        if self.current_message_widget:
            self.current_message_widget.append_content(chunk)
            QTimer.singleShot(50, self._scroll_to_bottom)
    
    def _on_tool_call_detected(self, tool_name: str):
        """Handle tool call detection."""
        tool_msg = MessageWidget(f"🔧 Executing tool: {tool_name}...", role="tool")
        container = self._wrap_message(tool_msg, "tool")
        
        # Track tool message container and set initial visibility
        self.tool_message_containers.append(container)
        container.setVisible(self.show_tool_messages)
        
        self.chat_layout.insertWidget(self.message_insert_index, container)
        self.message_insert_index += 1
        self._scroll_to_bottom()
    
    def _on_tool_result(self, tool_name: str, result: str):
        """Handle tool result."""
        # Try to format JSON nicely
        try:
            data = json.loads(result)
            formatted = json.dumps(data, indent=2)
            if len(formatted) > 500:
                formatted = formatted[:500] + "\n..."
        except:
            formatted = result[:500] + ("..." if len(result) > 500 else "")
        
        result_msg = MessageWidget(f"📋 {tool_name} result:\n{formatted}", role="tool")
        container = self._wrap_message(result_msg, "tool")
        
        # Track tool message container and set initial visibility
        self.tool_message_containers.append(container)
        container.setVisible(self.show_tool_messages)
        
        self.chat_layout.insertWidget(self.message_insert_index, container)
        self.message_insert_index += 1
        
        # Create new assistant message widget for the interpretation
        self.current_message_widget = MessageWidget("", role="assistant")
        assistant_container = self._wrap_message(self.current_message_widget, "assistant")
        self.chat_layout.insertWidget(self.message_insert_index, assistant_container)
        self.message_insert_index += 1
        
        self._scroll_to_bottom()
    
    def _on_finished(self, full_content: str):
        """Handle completion."""
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_content
        })
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.stop_button.hide()
        self.input_field.setFocus()
        
        # Schedule worker cleanup after signals are processed
        QTimer.singleShot(100, self._delayed_cleanup)
        self.current_message_widget = None
        
        self._scroll_to_bottom()
    
    def _delayed_cleanup(self):
        """Clean up worker after a delay to ensure thread has finished."""
        if self.current_worker and not self.current_worker.isRunning():
            self.current_worker.deleteLater()
            self.current_worker = None
    
    def _on_error(self, error: str):
        """Handle error."""
        error_msg = MessageWidget(f"❌ Error: {error}", role="assistant")
        error_msg.setStyleSheet("""
            MessageWidget {
                background: #fef2f2;
                border: 2px solid #ef4444;
                border-radius: 12px;
                padding: 12px;
                margin: 4px;
            }
        """)
        container = self._wrap_message(error_msg, "assistant")
        self.chat_layout.insertWidget(self.message_insert_index, container)
        self.message_insert_index += 1
        
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.stop_button.hide()
        
        # Schedule worker cleanup
        QTimer.singleShot(100, self._delayed_cleanup)
        self.current_message_widget = None
        
        self._scroll_to_bottom()
    
    def stop_request(self):
        """Stop current request."""
        if self.current_worker:
            self.current_worker.cancel()
            # Don't block - schedule cleanup
            QTimer.singleShot(100, self._delayed_cleanup)
        
        if self.current_message_widget:
            self.current_message_widget.append_content(" [Stopped]")
        
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.stop_button.hide()
        self.input_field.setFocus()
        
        self.current_message_widget = None
    
    def _scroll_to_bottom(self):
        """Scroll to bottom."""
        scrollbar = self.chat_scroll.verticalScrollBar()
        QApplication.processEvents()
        scrollbar.setValue(scrollbar.maximum())
    
    def init_status_bar(self):
        """Initialize status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background: #1e293b;
                color: #94a3b8;
                border-top: 1px solid #334155;
            }
        """)
        self.status_bar.showMessage("Initializing...")
    
    def update_connection_status(self):
        """Update connection status."""
        connected, msg = self.llm_client.check_connection()
        
        tools = self.mcp_manager.list_tools()
        tools_str = f" | Tools: {', '.join(tools)}" if tools else ""
        
        if connected:
            self.status_bar.showMessage(f"✓ Connected to {self.llm_client.model}{tools_str}")
            self.status_bar.setStyleSheet("""
                QStatusBar {
                    background: #065f46;
                    color: #d1fae5;
                    border-top: 1px solid #047857;
                }
            """)
        else:
            self.status_bar.showMessage(f"✗ Disconnected: {msg}")
            self.status_bar.setStyleSheet("""
                QStatusBar {
                    background: #991b1b;
                    color: #fecaca;
                    border-top: 1px solid #b91c1c;
                }
            """)
    
    def closeEvent(self, event):
        """Handle close."""
        # Stop connection timer first
        if hasattr(self, 'connection_timer'):
            self.connection_timer.stop()
        
        # Clean up worker thread properly
        self._cleanup_worker()
        
        event.accept()
