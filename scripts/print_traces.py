#!/usr/bin/env python3
"""
Print formatted agent traces from logs/agent_trace.jsonl (or any JSONL trace file).

Each line in the file is one trace produced by agent_search(), containing:
  - timestamp, query, system prompt
  - messages: user message, tool_call_item(s), tool_call_output_item(s), message_output_item(s)
  - output: final ProductRanking (product_ids)

Usage: python scripts/print_traces.py <file> <selector> [--full | --truncated]

Selector format:  trace.message
  "."     all traces, all messages
  "1."    trace 1, all messages
  ".2"    all traces, message 2
  "1.2"   trace 1, message 2

Examples:
  python scripts/print_traces.py logs/agent_trace.jsonl .
  python scripts/print_traces.py logs/agent_trace.jsonl 0.
  python scripts/print_traces.py logs/agent_trace.jsonl . --full
  python scripts/print_traces.py logs/agent_trace.jsonl . --truncated
"""

import argparse
import json
import sys
from typing import Any, Optional


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------

class C:
    RED    = '\033[91m'
    GREEN  = '\033[92m'
    CYAN   = '\033[96m'
    ORANGE = '\033[93m'
    RESET  = '\033[0m'
    BOLD   = '\033[1m'


# ---------------------------------------------------------------------------
# Message printing
# ---------------------------------------------------------------------------

def _get_type(msg: dict) -> str:
    return msg.get("type") or msg.get("role", "unknown")


def _format_tool_call(raw: dict) -> str:
    name = raw.get("name", "tool")
    args_str = raw.get("arguments", "{}")
    try:
        args = json.loads(args_str)
        args_fmt = ", ".join(f"{k}={v!r}" for k, v in args.items())
    except (json.JSONDecodeError, AttributeError):
        args_fmt = args_str
    return f"{name}({args_fmt})"


def _extract_assistant_text(raw: dict) -> str:
    content = raw.get("content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return " ".join(p for p in parts if p)
    return str(content) if content else ""


def print_message(
    msg: dict,
    idx: int,
    show_full: bool = False,
    show_truncated: bool = False,
) -> None:
    t = _get_type(msg)

    if t == "user":
        content = msg.get("content", "")
        print(f"{C.GREEN}{C.BOLD}[{idx}] USER:{C.RESET} {C.GREEN}{content}{C.RESET}")

    elif t == "tool_call_item":
        raw = msg.get("raw", {})
        call = _format_tool_call(raw)
        print(f"{C.CYAN}{C.BOLD}[{idx}] TOOL CALL:{C.RESET} {C.CYAN}{call}{C.RESET}")

    elif t == "tool_call_output_item":
        raw = msg.get("raw", {})
        output = raw.get("output", "")
        if show_full:
            print(f"{C.ORANGE}{C.BOLD}[{idx}] TOOL RESULT:{C.RESET} {C.ORANGE}{output}{C.RESET}")
        elif show_truncated:
            first_line = output.split("\n")[0]
            print(f"{C.ORANGE}{C.BOLD}[{idx}] TOOL RESULT:{C.RESET} {C.ORANGE}{first_line}{C.RESET}")
        else:
            # Default: show each product on one line, truncate long result
            lines = output.split("\n")
            preview = "\n  ".join(lines[:30])
            suffix = f"\n  ... ({len(lines) - 30} more)" if len(lines) > 30 else ""
            print(f"{C.ORANGE}{C.BOLD}[{idx}] TOOL RESULT:{C.RESET}\n{C.ORANGE}  {preview}{suffix}{C.RESET}")

    elif t == "message_output_item":
        raw = msg.get("raw", {})
        text = _extract_assistant_text(raw)
        print(f"{C.BOLD}[{idx}] ASSISTANT:{C.RESET} {text}")

    else:
        print(f"[{idx}] {t}: {msg}")


def print_trace(
    trace: dict,
    trace_idx: Optional[int] = None,
    message_idx: Optional[int] = None,
    show_full: bool = False,
    show_truncated: bool = False,
) -> None:
    label = f"Trace {trace_idx}" if trace_idx is not None else "Trace"
    ts = trace.get("timestamp", "")
    query = trace.get("query", "")
    output = trace.get("output", {})
    product_ids = output.get("product_ids", []) if isinstance(output, dict) else output

    print(f"\n{C.BOLD}{'=' * 70}{C.RESET}")
    print(f"{C.BOLD}{label}{C.RESET}  {ts}")
    print(f"{C.GREEN}{C.BOLD}Query:{C.RESET} {C.GREEN}{query}{C.RESET}")
    print(f"{C.BOLD}Output:{C.RESET} product_ids={product_ids}")
    print(f"{C.BOLD}{'-' * 70}{C.RESET}")

    messages = trace.get("messages", [])

    # Optionally print system prompt
    system = trace.get("system", "")
    if show_full and system:
        print(f"{C.RED}{C.BOLD}[sys] SYSTEM:{C.RESET} {C.RED}{system}{C.RESET}")

    if message_idx is not None:
        if 0 <= message_idx < len(messages):
            print_message(messages[message_idx], message_idx, show_full, show_truncated)
        else:
            print(
                f"Error: message index {message_idx} out of range "
                f"(0–{len(messages) - 1})",
                file=sys.stderr,
            )
    else:
        for i, msg in enumerate(messages):
            print_message(msg, i, show_full, show_truncated)


# ---------------------------------------------------------------------------
# Selector parsing
# ---------------------------------------------------------------------------

def parse_selector(selector: str) -> tuple[Optional[int], Optional[int]]:
    """Parse 'trace.message' selector. Returns (trace_idx, message_idx), None means all."""
    if selector == ".":
        return None, None
    parts = selector.split(".", 1)
    trace_idx = int(parts[0]) if parts[0] else None
    message_idx = int(parts[1]) if len(parts) > 1 and parts[1] else None
    return trace_idx, message_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_traces(filename: str) -> list[dict]:
    with open(filename) as f:
        content = f.read().strip()
    if not content:
        return []
    # Try JSON array first, fall back to JSONL
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        traces = []
        for i, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error on line {i}: {e}", file=sys.stderr)
                sys.exit(1)
        return traces


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print formatted agent traces from a JSONL trace file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("filename", help="Path to JSONL trace file")
    parser.add_argument(
        "selector",
        nargs="?",
        default=".",
        help="trace.message selector (default: '.' = all traces, all messages)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", help="Show full tool outputs and system prompt")
    group.add_argument("--truncated", action="store_true", help="Show only first line of tool outputs")
    args = parser.parse_args()

    traces = load_traces(args.filename)
    if not traces:
        print("No traces found.")
        return

    try:
        trace_idx, message_idx = parse_selector(args.selector)
    except ValueError as e:
        print(f"Invalid selector: {e}", file=sys.stderr)
        sys.exit(1)

    if trace_idx is None:
        indices = range(len(traces))
    else:
        if trace_idx >= len(traces):
            print(f"Error: trace index {trace_idx} out of range (0–{len(traces) - 1})", file=sys.stderr)
            sys.exit(1)
        indices = [trace_idx]

    if len(indices) > 1:
        print(f"{C.RED}{C.BOLD}{'=' * 70}")
        print(f"{len(traces)} traces in {args.filename}{C.RESET}")

    for i in indices:
        print_trace(traces[i], i, message_idx, args.full, args.truncated)


if __name__ == "__main__":
    main()
