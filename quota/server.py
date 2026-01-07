#!/usr/bin/env python3
"""MCP Server for disk quota management."""

import subprocess
import json
import re
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("quota")


def parse_quota_output(output: str) -> dict:
    """Parse the quota command output into structured data."""
    result = {
        "user_quotas": [],
        "slate_projects": []
    }
    
    lines = output.strip().split("\n")
    current_section = "user"
    
    # Skip header lines
    in_data = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Detect section change
        if "Slate Projects" in line:
            current_section = "projects"
            continue
        
        # Skip header line
        if line.startswith("Disk quotas") or line.startswith("Filesystem"):
            continue
            
        # Skip footnote
        if line.startswith("*"):
            continue
        
        # Remove ANSI color codes
        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        
        # Parse data lines
        # Format: name  usage  quota  [progress bar]  timestamp
        parts = clean_line.split()
        if len(parts) >= 3:
            name = parts[0]
            
            # Check if this is a "files" entry (second word is "files")
            is_files = False
            if len(parts) >= 4 and parts[1] == "files":
                is_files = True
                usage = parts[2]
                quota = parts[3]
            else:
                usage = parts[1]
                quota = parts[2]
            
            # Extract percentage from the progress bar if present
            percent_match = re.search(r'(\d+)%', clean_line)
            percent = int(percent_match.group(1)) if percent_match else None
            
            entry = {
                "name": name,
                "type": "files" if is_files else "storage",
                "usage": usage,
                "quota": quota,
                "percent_used": percent
            }
            
            if current_section == "user":
                result["user_quotas"].append(entry)
            else:
                result["slate_projects"].append(entry)
    
    return result


@mcp.tool()
def get_quota() -> str:
    """Get current disk quota information for the user.
    
    Returns quota information for all filesystems including home, scratch,
    slate storage, and any Slate Projects the user has access to.
    
    The output includes both storage quotas (in GB/TB) and file count quotas,
    along with current usage and percentage used.
    """
    try:
        result = subprocess.run(
            ["quota"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return json.dumps({
                "error": f"quota command failed: {result.stderr.strip()}"
            })
        
        parsed = parse_quota_output(result.stdout)
        
        return json.dumps({
            "user": os.environ.get("USER", os.environ.get("LOGNAME", "unknown")),
            "quotas": parsed,
            "raw_output": result.stdout
        }, indent=2)
        
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "quota command timed out"})
    except FileNotFoundError:
        return json.dumps({"error": "quota command not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_storage_quota() -> str:
    """Get only the storage (space) quota information.
    
    Returns storage quota information showing disk space usage in GB/TB.
    This is the amount of data you can store on each filesystem.
    
    Storage quota differs from file quota:
    - Storage quota limits how much total data (in bytes) you can store
    - A few large files can fill your storage quota quickly
    - Measured in GB or TB
    """
    try:
        result = subprocess.run(
            ["quota"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return json.dumps({
                "error": f"quota command failed: {result.stderr.strip()}"
            })
        
        parsed = parse_quota_output(result.stdout)
        
        # Filter to only storage quotas (not files)
        storage_quotas = {
            "user_quotas": [q for q in parsed["user_quotas"] if q["type"] == "storage"],
            "slate_projects": [q for q in parsed["slate_projects"] if q["type"] == "storage"]
        }
        
        return json.dumps({
            "user": os.environ.get("USER", os.environ.get("LOGNAME", "unknown")),
            "storage_quotas": storage_quotas
        }, indent=2)
        
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "quota command timed out"})
    except FileNotFoundError:
        return json.dumps({"error": "quota command not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_file_quota() -> str:
    """Get only the file count quota information.
    
    Returns file quota information showing the number of files/inodes.
    This is the total number of files and directories you can create.
    
    File quota differs from storage quota:
    - File quota (inode quota) limits the NUMBER of files you can have
    - Many small files can exhaust your file quota even with storage space left
    - Measured in number of files/inodes
    - Common issue: extracting archives with millions of small files
    """
    try:
        result = subprocess.run(
            ["quota"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return json.dumps({
                "error": f"quota command failed: {result.stderr.strip()}"
            })
        
        parsed = parse_quota_output(result.stdout)
        
        # Filter to only file quotas
        file_quotas = {
            "user_quotas": [q for q in parsed["user_quotas"] if q["type"] == "files"],
            "slate_projects": [q for q in parsed["slate_projects"] if q["type"] == "files"]
        }
        
        return json.dumps({
            "user": os.environ.get("USER", os.environ.get("LOGNAME", "unknown")),
            "file_quotas": file_quotas
        }, indent=2)
        
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "quota command timed out"})
    except FileNotFoundError:
        return json.dumps({"error": "quota command not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def explain_quota_difference() -> str:
    """Explain the difference between storage quota and file quota.
    
    Returns a detailed explanation of the two types of quotas and
    common scenarios where each might be the limiting factor.
    """
    explanation = {
        "storage_quota": {
            "description": "Storage quota limits the total amount of data (in bytes) you can store.",
            "unit": "GB or TB",
            "common_issues": [
                "Large data files (genomics, simulations, images)",
                "Uncompressed datasets",
                "Accumulated output files from many jobs"
            ],
            "solutions": [
                "Compress files with gzip, bzip2, or xz",
                "Move old data to archive storage",
                "Delete temporary and intermediate files"
            ]
        },
        "file_quota": {
            "description": "File quota (inode quota) limits the total number of files and directories you can create.",
            "unit": "Number of files/inodes",
            "common_issues": [
                "Extracting archives with many small files",
                "Node.js node_modules directories",
                "Python virtual environments",
                "Git repositories with long history",
                "Conda environments"
            ],
            "solutions": [
                "Use containerized environments (Singularity/Apptainer)",
                "Combine small files into archives (tar)",
                "Clean up package caches and temporary files",
                "Use shared conda installations when available"
            ]
        },
        "key_difference": "You can run out of file quota while having plenty of storage space left (many small files), or run out of storage while having file quota available (few large files)."
    }
    
    return json.dumps(explanation, indent=2)


if __name__ == "__main__":
    mcp.run()

