"""Pact CLI — entrypoint for scoring environment and development.

Usage:
  pact run "task text"          Run a single task
  pact batch tasks.jsonl        Run tasks from JSONL file
  pact serve                    Start HTTP API
  pact bench                    Run benchmark comparison (mock)
  pact eval < task.json         Read from stdin, output result to stdout
"""

from __future__ import annotations

import json
import logging
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

from .router import PactRouter


def _setup_logging():
    level = os.getenv("PACT_LOG", "WARNING").upper()
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, level, logging.WARNING),
        format="%(levelname)s [%(name)s] %(message)s",
    )


def cmd_run(args):
    """Run a single task."""
    router = PactRouter()
    task = args.task
    if not task:
        task = sys.stdin.read().strip()
    if not task:
        print("Usage: pact run \"task text\" or pipe task to stdin", file=sys.stderr)
        sys.exit(1)

    result = router.process(task)
    print(json.dumps(result, indent=2))
    _print_summary(result)


def cmd_eval(args):
    """Read task from stdin, output JSON result to stdout."""
    router = PactRouter()
    try:
        data = json.load(sys.stdin)
        task = data.get("task", data.get("text", ""))
    except json.JSONDecodeError:
        sys.stdin.seek(0)
        task = sys.stdin.read().strip()

    if not task:
        print(json.dumps({"error": "no task provided"}), file=sys.stderr)
        sys.exit(1)

    result = router.process(task)
    result.pop("_trace", None)  # Keep eval output clean
    print(json.dumps(result))


def cmd_batch(args):
    """Run tasks from a JSONL file."""
    router = PactRouter()
    tasks = []
    with open(args.file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    tasks.append(obj.get("task", obj.get("text", line)))
                except json.JSONDecodeError:
                    tasks.append(line)

    results = router.process_batch(tasks, show_trace=args.trace)
    out = sys.stdout
    for r in results:
        if not args.trace:
            r.pop("_trace", None)
        out.write(json.dumps(r) + "\n")

    _print_batch_summary(results)


def cmd_serve(args):
    """Start HTTP API server for scoring integration."""
    router = PactRouter()
    port = args.port

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._reply(400, {"error": "invalid json"})
                return

            task = data.get("task", data.get("text", ""))
            if not task:
                self._reply(400, {"error": "no task in request"})
                return

            result = router.process(task)
            result.pop("_trace", None)
            self._reply(200, result)

        def _reply(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, fmt, *args):
            logger = logging.getLogger("pact.http")
            logger.debug(fmt % args)

    server = HTTPServer(("0.0.0.0", port), Handler)
    logging.getLogger("pact.http").info("Serving on :%d", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def cmd_bench(args):
    """Run a benchmark comparing Pact cascade vs all-Fireworks baseline."""
    from . import bench
    bench.run()


def _print_summary(result: dict):
    p = result.get("pact", {})
    print(file=sys.stderr)
    print(f"  Route:       {p.get('route', '?')}", file=sys.stderr)
    print(f"  Difficulty:  {p.get('difficulty', '?')}", file=sys.stderr)
    print(f"  Escalations: {p.get('escalated', 0)}", file=sys.stderr)
    print(f"  FW tokens:   {p.get('fireworks_tokens', 0)}", file=sys.stderr)
    print(f"  Time:        {p.get('elapsed_ms', 0)}ms", file=sys.stderr)


def _print_batch_summary(results: list):
    if not results:
        return
    total_tokens = sum(r.get("pact", {}).get("fireworks_tokens", 0) for r in results)
    total_escalations = sum(r.get("pact", {}).get("escalated", 0) for r in results)
    local_count = sum(1 for r in results if r.get("pact", {}).get("route") == "local")
    fw_count = len(results) - local_count
    print(file=sys.stderr)
    print(f"  Total tasks: {len(results)}", file=sys.stderr)
    print(f"  Local:       {local_count} | Fireworks: {fw_count}", file=sys.stderr)
    print(f"  Total FW tokens: {total_tokens}", file=sys.stderr)
    print(f"  Total escalations: {total_escalations}", file=sys.stderr)


def main():
    _setup_logging()

    import argparse
    parser = argparse.ArgumentParser(
        prog="pact",
        description="Pact — Protocol for Agent Compact Transfer.\n"
                    "Hybrid token-efficient routing agent for AMD Hackathon ACT II.",
    )
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", help="Run a single task")
    p_run.add_argument("task", nargs="?", help="Task text (omit to read stdin)")

    sub.add_parser("eval", help="Read task JSON from stdin, output result to stdout")

    p_batch = sub.add_parser("batch", help="Run tasks from JSONL file")
    p_batch.add_argument("file", help="JSONL file path")
    p_batch.add_argument("--trace", action="store_true", help="Include PACT signal trace in output")

    p_serve = sub.add_parser("serve", help="Start HTTP API server")
    p_serve.add_argument("--port", "-p", type=int, default=int(os.getenv("PACT_PORT", "8080")))

    sub.add_parser("bench", help="Run benchmark (mock comparison)")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "bench":
        cmd_bench(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
