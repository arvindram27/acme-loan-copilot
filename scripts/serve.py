"""Run the Acme Home Loans copilot web app (milestone 8).

    python scripts/serve.py [--port 8730]
"""

import argparse

import uvicorn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8730)
    args = ap.parse_args()
    uvicorn.run("agentmem.app.main:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
