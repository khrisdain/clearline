"""Command line: load a folder of documents, ask a question, or serve the API.

    clearline ingest sample_data
    clearline ask "How many days of telework a week?" --as ben
    clearline serve
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .auth import DEV_USERS
from .config import get_settings
from .ingest import load_directory
from .service import Clearline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="clearline")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Load .md, .txt and .pdf files from a folder")
    ingest.add_argument("path", type=Path)
    ask = sub.add_parser("ask", help="Ask a question as one of the dev users (loads sample_data first with --load)")
    ask.add_argument("question")
    ask.add_argument("--as", dest="user", default="ana", choices=sorted(DEV_USERS))
    ask.add_argument("--load", type=Path, help="Folder to ingest first (for the in-memory store)")
    serve = sub.add_parser("serve", help="Run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("clearline.main:app", host=args.host, port=args.port)
        return
    clearline = Clearline(get_settings())
    if args.command == "ingest" or getattr(args, "load", None):
        for document in load_directory(args.path if args.command == "ingest" else args.load):
            out = clearline.ingest(document, "cli")
            if args.command == "ingest":
                print(f"{out.classification:<13} {out.chunks:>3} chunks  {out.title}")
    if args.command == "ask":
        answer = clearline.ask(args.question, DEV_USERS[args.user])
        print(json.dumps(answer.model_dump(), indent=2))


if __name__ == "__main__":
    main()
