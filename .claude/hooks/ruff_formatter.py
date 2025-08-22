#!/usr/bin/env python3
"""
Multi-format formatter for Claude Code output.
Automatically formats Python files with ruff, TOML files with taplo, and Markdown files with mdformat.
"""

import json
import os
import subprocess
import sys


def run_ruff(file_path, command_args):
    """Run ruff command on a file."""
    # Try to find ruff in various locations
    ruff_paths = [
        "/Users/minpeter/github.com/friendliai/aws-hackday-micro-internal/.venv/bin/ruff",
        "ruff",
    ]

    ruff_cmd = None
    for path in ruff_paths:
        try:
            subprocess.run([path, "--version"], capture_output=True, check=False)
            ruff_cmd = path
            break
        except (FileNotFoundError, OSError):
            continue

    if not ruff_cmd:
        print("Warning: ruff not found. Please install with: uv pip install ruff", file=sys.stderr)
        return False, "", "ruff not found"

    try:
        result = subprocess.run(
            [ruff_cmd, *command_args, file_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(file_path) or ".",
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def run_command(cmd_path_candidates, version_arg, file_path, command_args, tool_name):
    """Generic function to run a command-line tool."""
    cmd = None
    for path in cmd_path_candidates:
        try:
            subprocess.run([path, version_arg], capture_output=True, check=False)
            cmd = path
            break
        except (FileNotFoundError, OSError):
            continue

    if not cmd:
        print(f"Warning: {tool_name} not found. Please install it.", file=sys.stderr)
        return False, "", f"{tool_name} not found"

    try:
        result = subprocess.run(
            [cmd] + command_args + [file_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(file_path) or ".",
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def format_and_lint_python(file_path):
    """Format and lint Python file with ruff."""
    if not os.path.exists(file_path):
        return

    # First, format the file
    format_success, format_out, format_err = run_ruff(file_path, ["format"])
    if format_success:
        if format_out:
            print(f"✓ Formatted {file_path}")
    else:
        if format_err:
            print(f"⚠ Format warning for {file_path}: {format_err}", file=sys.stderr)

    # Then, fix linting issues
    fix_success, fix_out, fix_err = run_ruff(file_path, ["check", "--fix"])
    if fix_success:
        if fix_out:
            print(f"✓ Fixed linting issues in {file_path}")
    else:
        # Run check without fix to get remaining issues
        check_success, check_out, check_err = run_ruff(file_path, ["check"])
        if not check_success and check_out:
            print(f"⚠ Linting issues in {file_path}:\n{check_out}", file=sys.stderr)


def format_toml(file_path):
    """Format TOML file with taplo."""
    if not os.path.exists(file_path):
        return

    taplo_candidates = ["taplo", "uv run taplo"]
    success, stdout, stderr = run_command(
        taplo_candidates, "--version", file_path, ["format"], "taplo"
    )

    if success:
        print(f"✓ Formatted TOML file {file_path}")
    else:
        if stderr:
            print(f"⚠ TOML format warning for {file_path}: {stderr}", file=sys.stderr)


def format_markdown(file_path):
    """Format Markdown file with mdformat."""
    if not os.path.exists(file_path):
        return

    mdformat_candidates = ["uv run mdformat", "mdformat"]
    # Use a wrapper since we need to handle uv run specially
    try:
        result = subprocess.run(
            ["uv", "run", "mdformat", "--wrap", "100", file_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(file_path) or ".",
        )

        if result.returncode == 0:
            print(f"✓ Formatted Markdown file {file_path}")
        else:
            if result.stderr:
                print(
                    f"⚠ Markdown format warning for {file_path}: {result.stderr}", file=sys.stderr
                )
    except Exception as e:
        print(f"⚠ Error formatting Markdown file {file_path}: {e}", file=sys.stderr)


# Main execution
try:
    input_data = json.load(sys.stdin)
    file_path = input_data.get("tool_input", {}).get("file_path", "")

    # Handle multiple files (for MultiEdit)
    if "edits" in input_data.get("tool_input", {}):
        # For MultiEdit, file_path is in the root of tool_input
        if file_path:
            if file_path.endswith(".py"):
                format_and_lint_python(file_path)
            elif file_path.endswith(".toml"):
                format_toml(file_path)
            elif file_path.endswith(".md"):
                format_markdown(file_path)
    # Handle single file (for Edit/Write)
    elif file_path:
        if file_path.endswith(".py"):
            format_and_lint_python(file_path)
        elif file_path.endswith(".toml"):
            format_toml(file_path)
        elif file_path.endswith(".md"):
            format_markdown(file_path)

except Exception as e:
    print(f"Error in formatter hook: {e}", file=sys.stderr)
    sys.exit(1)
