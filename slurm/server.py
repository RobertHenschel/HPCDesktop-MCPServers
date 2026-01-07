#!/usr/bin/env python3
"""MCP Server for Slurm partition management."""

import subprocess
import json
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slurm")


@mcp.tool()
def list_partitions() -> str:
    """List all available Slurm partitions.
    
    Returns partition information including name, availability, time limit,
    and total nodes.
    """
    try:
        # Run sinfo to get partition information
        result = subprocess.run(
            ["sinfo", "--format=%P|%a|%l|%D", "--noheader"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return json.dumps({
                "error": f"sinfo command failed: {result.stderr.strip()}"
            })
        
        # Aggregate partitions by name
        partition_map = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 4:
                partition_name = parts[0].rstrip("*")
                is_default = parts[0].endswith("*")
                node_count = int(parts[3]) if parts[3].isdigit() else 0
                
                if partition_name not in partition_map:
                    partition_map[partition_name] = {
                        "name": partition_name,
                        "default": is_default,
                        "availability": parts[1],
                        "time_limit": parts[2],
                        "total_nodes": 0
                    }
                
                partition_map[partition_name]["total_nodes"] += node_count
        
        partitions = list(partition_map.values())
        
        return json.dumps({
            "partitions": partitions,
            "count": len(partitions)
        }, indent=2)
        
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "sinfo command timed out"})
    except FileNotFoundError:
        return json.dumps({"error": "sinfo command not found. Is Slurm installed?"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_my_jobs() -> str:
    """Get all running and pending jobs for the current user.
    
    Returns job information including job ID, name, state, partition,
    time used, number of nodes, and node list.
    """
    try:
        user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
        if not user:
            return json.dumps({"error": "Could not determine current user"})
        
        # Run squeue to get user's jobs, filtering for running (R) and pending (PD) states
        result = subprocess.run(
            [
                "squeue",
                f"--user={user}",
                "--states=RUNNING,PENDING",
                "--format=%i|%j|%T|%P|%M|%D|%N|%r",
                "--noheader"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return json.dumps({
                "error": f"squeue command failed: {result.stderr.strip()}"
            })
        
        jobs = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 8:
                jobs.append({
                    "job_id": parts[0].strip(),
                    "name": parts[1].strip(),
                    "state": parts[2].strip(),
                    "partition": parts[3].strip(),
                    "time_used": parts[4].strip(),
                    "nodes": parts[5].strip(),
                    "nodelist": parts[6].strip(),
                    "reason": parts[7].strip()
                })
        
        return json.dumps({
            "user": user,
            "jobs": jobs,
            "count": len(jobs)
        }, indent=2)
        
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "squeue command timed out"})
    except FileNotFoundError:
        return json.dumps({"error": "squeue command not found. Is Slurm installed?"})
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp.run()

