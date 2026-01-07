#!/usr/bin/env python3
"""MCP Server Viewer - A Qt application to display available MCP servers and their functions."""

import sys
import os
import json
import ast
import re
import warnings
from pathlib import Path

# Suppress PyQt6/SIP internal deprecation warning
warnings.filterwarnings("ignore", message="sipPyTypeDict.*deprecated")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTreeWidget, QTreeWidgetItem, QLabel, QFrame, QTextEdit,
    QSplitter, QHeaderView, QPushButton, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap
import importlib.util


def extract_mcp_functions(python_file: str) -> list[dict]:
    """Parse a Python file and extract functions decorated with @mcp.tool()."""
    functions = []
    
    try:
        with open(python_file, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has @mcp.tool() decorator
                for decorator in node.decorator_list:
                    is_mcp_tool = False
                    
                    # Handle @mcp.tool() or @mcp.tool
                    if isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Attribute):
                            if decorator.func.attr == 'tool':
                                is_mcp_tool = True
                    elif isinstance(decorator, ast.Attribute):
                        if decorator.attr == 'tool':
                            is_mcp_tool = True
                    
                    if is_mcp_tool:
                        # Extract docstring
                        docstring = ast.get_docstring(node) or "No description available"
                        
                        # Extract parameters
                        params = []
                        for arg in node.args.args:
                            param_name = arg.arg
                            param_type = ""
                            if arg.annotation:
                                param_type = ast.unparse(arg.annotation)
                            params.append({"name": param_name, "type": param_type})
                        
                        # Extract return type
                        return_type = ""
                        if node.returns:
                            return_type = ast.unparse(node.returns)
                        
                        functions.append({
                            "name": node.name,
                            "docstring": docstring,
                            "parameters": params,
                            "return_type": return_type,
                            "line": node.lineno
                        })
                        break
                        
    except Exception as e:
        functions.append({
            "name": "ERROR",
            "docstring": f"Failed to parse file: {e}",
            "parameters": [],
            "return_type": "",
            "line": 0
        })
    
    return functions


class ExecutionWorker(QThread):
    """Worker thread to execute MCP functions without blocking the UI."""
    finished = pyqtSignal(str, bool)  # result, is_error
    
    def __init__(self, mcp_path: str, function_name: str):
        super().__init__()
        self.mcp_path = mcp_path
        self.function_name = function_name
    
    def run(self):
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("mcp_module", self.mcp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get and call the function
            func = getattr(module, self.function_name)
            result = func()
            
            # Try to pretty-print JSON
            try:
                parsed = json.loads(result)
                result = json.dumps(parsed, indent=2)
            except (json.JSONDecodeError, TypeError):
                pass
            
            self.finished.emit(result, False)
        except Exception as e:
            self.finished.emit(f"Error executing function: {e}", True)


class ResultDialog(QDialog):
    """Modal dialog to display function execution results."""
    
    def __init__(self, parent, function_name: str, result: str, is_error: bool = False):
        super().__init__(parent)
        self.setWindowTitle(f"Result: {function_name}()")
        self.setMinimumSize(600, 400)
        self.setModal(True)
        
        # Inherit parent's icon
        if parent.windowIcon():
            self.setWindowIcon(parent.windowIcon())
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header
        header_label = QLabel(f"⚡ {function_name}()")
        header_label.setFont(QFont("JetBrains Mono", 16, QFont.Weight.Bold))
        header_label.setStyleSheet("color: #00d9ff;" if not is_error else "color: #ff6b6b;")
        layout.addWidget(header_label)
        
        # Result text
        result_text = QTextEdit()
        result_text.setReadOnly(True)
        result_text.setFont(QFont("JetBrains Mono", 11))
        result_text.setPlainText(result)
        result_text.setStyleSheet("""
            QTextEdit {
                background-color: rgba(26, 26, 46, 0.95);
                border: 1px solid #e94560;
                border-radius: 8px;
                color: #98c379;
                padding: 12px;
            }
        """ if not is_error else """
            QTextEdit {
                background-color: rgba(26, 26, 46, 0.95);
                border: 1px solid #ff6b6b;
                border-radius: 8px;
                color: #ff6b6b;
                padding: 12px;
            }
        """)
        layout.addWidget(result_text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFont(QFont("JetBrains Mono", 11))
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #e94560;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 30px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6b8a;
            }
            QPushButton:pressed {
                background-color: #c73e54;
            }
        """)
        close_btn.clicked.connect(self.accept)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)


class MCPViewerWindow(QMainWindow):
    """Main window for the MCP Server Viewer application."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MCP Server Viewer")
        self.setMinimumSize(900, 600)
        
        # Get the directory where this script is located
        self.base_dir = Path(__file__).parent.resolve()
        
        # Set window icon for Alt-Tab and taskbar
        icon_path = self.base_dir / "Cray-1-AI.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        self.setup_style()
        self.setup_ui()
        self.load_mcps()
    
    def setup_style(self):
        """Apply custom styling to the application."""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
            QTreeWidget {
                background-color: rgba(26, 26, 46, 0.9);
                border: 1px solid #e94560;
                border-radius: 8px;
                color: #eaeaea;
                font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
                font-size: 13px;
                padding: 8px;
            }
            QTreeWidget::item {
                padding: 6px 4px;
                border-radius: 4px;
            }
            QTreeWidget::item:hover {
                background-color: rgba(233, 69, 96, 0.2);
            }
            QTreeWidget::item:selected {
                background-color: rgba(233, 69, 96, 0.4);
                color: #ffffff;
            }
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {
                border-image: none;
                image: none;
            }
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {
                border-image: none;
                image: none;
            }
            QHeaderView::section {
                background-color: #0f3460;
                color: #e94560;
                padding: 8px;
                border: none;
                font-weight: bold;
                font-family: 'JetBrains Mono', monospace;
            }
            QTextEdit {
                background-color: rgba(26, 26, 46, 0.95);
                border: 1px solid #e94560;
                border-radius: 8px;
                color: #eaeaea;
                font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
                font-size: 12px;
                padding: 12px;
            }
            QLabel {
                color: #eaeaea;
            }
            QSplitter::handle {
                background-color: #e94560;
                width: 2px;
            }
            QFrame#header {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(233, 69, 96, 0.3), stop:1 rgba(15, 52, 96, 0.3));
                border-radius: 10px;
                padding: 10px;
            }
        """)
    
    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        
        # Try to load logo
        logo_path = self.base_dir / "Cray-1-AI.png"
        if logo_path.exists():
            logo_label = QLabel()
            pixmap = QPixmap(str(logo_path))
            scaled_pixmap = pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, 
                                          Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            header_layout.addWidget(logo_label)
        
        title_layout = QVBoxLayout()
        title_label = QLabel("MCP Server Viewer")
        title_label.setFont(QFont("JetBrains Mono", 24, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #e94560;")
        
        subtitle_label = QLabel("Model Context Protocol Server Explorer")
        subtitle_label.setFont(QFont("JetBrains Mono", 11))
        subtitle_label.setStyleSheet("color: #7f8c8d;")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        main_layout.addWidget(header)
        
        # Splitter for tree and details
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Tree widget for MCP servers
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["MCP Servers & Functions"])
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tree.setAnimated(True)
        self.tree.setIndentation(25)
        self.tree.itemClicked.connect(self.on_item_clicked)
        
        # Details panel
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        
        details_header_layout = QHBoxLayout()
        details_title = QLabel("Details")
        details_title.setFont(QFont("JetBrains Mono", 14, QFont.Weight.Bold))
        details_title.setStyleSheet("color: #e94560; margin-bottom: 5px;")
        details_header_layout.addWidget(details_title)
        details_header_layout.addStretch()
        
        # Execute button
        self.execute_btn = QPushButton("▶ Execute")
        self.execute_btn.setFont(QFont("JetBrains Mono", 11, QFont.Weight.Bold))
        self.execute_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.execute_btn.setEnabled(False)
        self.execute_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d9ff;
                color: #1a1a2e;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #33e5ff;
            }
            QPushButton:pressed {
                background-color: #00b8d9;
            }
            QPushButton:disabled {
                background-color: #4a4a5a;
                color: #7a7a8a;
            }
        """)
        self.execute_btn.clicked.connect(self.execute_function)
        details_header_layout.addWidget(self.execute_btn)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setPlaceholderText("Select an MCP server or function to view details...")
        
        details_layout.addLayout(details_header_layout)
        details_layout.addWidget(self.details_text)
        
        # Track currently selected function
        self.selected_function = None
        self.worker = None
        
        splitter.addWidget(self.tree)
        splitter.addWidget(details_widget)
        splitter.setSizes([400, 500])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        main_layout.addWidget(self.status_label)
    
    def load_mcps(self):
        """Load MCP servers from the JSON configuration file."""
        config_path = self.base_dir / "available_mcps.json"
        
        if not config_path.exists():
            self.status_label.setText(f"⚠️  Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            self.status_label.setText(f"⚠️  Error parsing config: {e}")
            return
        
        mcps = config.get("mcps", [])
        total_functions = 0
        
        for mcp in mcps:
            mcp_name = mcp.get("name", "Unknown")
            mcp_desc = mcp.get("description", "")
            mcp_path = mcp.get("path", "")
            
            full_path = self.base_dir / mcp_path
            
            # Create MCP server item
            mcp_item = QTreeWidgetItem(self.tree)
            mcp_item.setText(0, f"🖥️  {mcp_name}")
            mcp_item.setFont(0, QFont("JetBrains Mono", 12, QFont.Weight.Bold))
            mcp_item.setForeground(0, QColor("#e94560"))
            mcp_item.setData(0, Qt.ItemDataRole.UserRole, {
                "type": "mcp",
                "name": mcp_name,
                "description": mcp_desc,
                "path": str(full_path)
            })
            
            # Extract and add functions
            if full_path.exists():
                functions = extract_mcp_functions(str(full_path))
                total_functions += len(functions)
                
                for func in functions:
                    func_item = QTreeWidgetItem(mcp_item)
                    func_item.setText(0, f"⚡ {func['name']}()")
                    func_item.setFont(0, QFont("JetBrains Mono", 11))
                    func_item.setForeground(0, QColor("#00d9ff"))
                    func_item.setData(0, Qt.ItemDataRole.UserRole, {
                        "type": "function",
                        "mcp_name": mcp_name,
                        "mcp_path": str(full_path),
                        **func
                    })
            else:
                error_item = QTreeWidgetItem(mcp_item)
                error_item.setText(0, f"⚠️  File not found: {mcp_path}")
                error_item.setForeground(0, QColor("#ff6b6b"))
            
            mcp_item.setExpanded(True)
        
        self.status_label.setText(
            f"✓ Loaded {len(mcps)} MCP server(s) with {total_functions} function(s)"
        )
    
    def on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item click to show details."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        
        # Track selected function and update execute button
        if data["type"] == "function":
            self.selected_function = data
            self.execute_btn.setEnabled(True)
        else:
            self.selected_function = None
            self.execute_btn.setEnabled(False)
        
        if data["type"] == "mcp":
            html = f"""
            <h2 style="color: #e94560; margin-bottom: 10px;">{data['name']}</h2>
            <p style="color: #aaa; margin-bottom: 15px;">{data['description']}</p>
            <p><b style="color: #00d9ff;">Path:</b> <code style="color: #98c379;">{data['path']}</code></p>
            """
        else:
            params_html = ""
            if data["parameters"]:
                params_html = "<ul style='margin: 5px 0; padding-left: 20px;'>"
                for p in data["parameters"]:
                    type_str = f": <span style='color: #e5c07b;'>{p['type']}</span>" if p['type'] else ""
                    params_html += f"<li><code style='color: #61afef;'>{p['name']}</code>{type_str}</li>"
                params_html += "</ul>"
            else:
                params_html = "<p style='color: #666; margin-left: 10px;'>No parameters</p>"
            
            return_html = f"<code style='color: #e5c07b;'>{data['return_type']}</code>" if data['return_type'] else "<span style='color: #666;'>None</span>"
            
            html = f"""
            <h2 style="color: #00d9ff; margin-bottom: 5px;">{data['name']}()</h2>
            <p style="color: #666; font-size: 11px; margin-bottom: 15px;">from {data['mcp_name']} • line {data['line']}</p>
            
            <h3 style="color: #e94560; margin-top: 15px;">Description</h3>
            <p style="color: #ccc; white-space: pre-wrap; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;">{data['docstring']}</p>
            
            <h3 style="color: #e94560; margin-top: 15px;">Parameters</h3>
            {params_html}
            
            <h3 style="color: #e94560; margin-top: 15px;">Returns</h3>
            <p style="margin-left: 10px;">{return_html}</p>
            """
        
        self.details_text.setHtml(html)
    
    def execute_function(self):
        """Execute the currently selected MCP function."""
        if not self.selected_function:
            return
        
        mcp_path = self.selected_function.get("mcp_path")
        func_name = self.selected_function.get("name")
        
        if not mcp_path or not func_name:
            return
        
        # Disable button during execution
        self.execute_btn.setEnabled(False)
        self.execute_btn.setText("⏳ Running...")
        self.status_label.setText(f"Executing {func_name}()...")
        
        # Run in background thread
        self.worker = ExecutionWorker(mcp_path, func_name)
        self.worker.finished.connect(self.on_execution_finished)
        self.worker.start()
    
    def on_execution_finished(self, result: str, is_error: bool):
        """Handle completion of function execution."""
        # Re-enable button
        self.execute_btn.setEnabled(True)
        self.execute_btn.setText("▶ Execute")
        
        func_name = self.selected_function.get("name", "function") if self.selected_function else "function"
        
        if is_error:
            self.status_label.setText(f"⚠️  Error executing {func_name}()")
        else:
            self.status_label.setText(f"✓ Successfully executed {func_name}()")
        
        # Show result dialog
        dialog = ResultDialog(self, func_name, result, is_error)
        dialog.exec()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set application-wide dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(26, 26, 46))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(234, 234, 234))
    palette.setColor(QPalette.ColorRole.Base, QColor(26, 26, 46))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(22, 33, 62))
    palette.setColor(QPalette.ColorRole.Text, QColor(234, 234, 234))
    palette.setColor(QPalette.ColorRole.Button, QColor(15, 52, 96))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(234, 234, 234))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(233, 69, 96))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = MCPViewerWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

