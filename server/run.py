#!/usr/bin/env python3
"""Run TBVoice server locally (without Docker)."""

import sys
from pathlib import Path

# Ensure project root is on path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uvicorn  # noqa: E402

from server.app import config  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "server.app.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        reload=config.DEBUG,
    )
