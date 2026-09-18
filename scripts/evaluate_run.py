"""Backward-compatible entry point for ``python scripts/evaluate_run.py``.

The implementation now lives in :mod:`model_doctor.app.evaluate_run`, so that an
installed wheel carries the evaluation stage rather than depending on this
directory, which is part of the checkout and not of the distribution.

This file exists only so the command documented in the README and used from a
source checkout keeps working. It adds no behaviour: it forwards ``argv`` to
the module's ``main`` and returns its exit status.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_doctor.app.evaluate_run import main

if __name__ == "__main__":
    raise SystemExit(main())
