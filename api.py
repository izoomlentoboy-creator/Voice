"""DEPRECATED — use ``server.app.main:app`` instead.

    uvicorn server.app.main:app --host 0.0.0.0 --port 8000

This file is kept only for backwards compatibility and will be removed
in a future release.
"""

import warnings

warnings.warn(
    "api.py is deprecated. Use 'uvicorn server.app.main:app' instead.",
    DeprecationWarning,
    stacklevel=1,
)

from server.app.main import app  # noqa: F401 — re-export for uvicorn
