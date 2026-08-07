"""Centralised logger construction.

Every module obtains its logger through :func:`get_logger` rather than calling
``logging.basicConfig`` itself. Libraries that configure the root logger fight
each other and produce duplicated or missing output; doing it in exactly one
place avoids that class of bug entirely.
"""

from __future__ import annotations

import logging
import sys

import config

_CONFIGURED = False


def _configure_root_once() -> None:
    """Attach a single stdout handler to the root logger, exactly once.

    Guarded by a module-level flag because ``get_logger`` is called at import
    time by several modules, and adding a handler per call would make every log
    line appear N times.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(fmt=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    )

    root = logging.getLogger()
    root.setLevel(config.LOG_LEVEL)
    root.addHandler(handler)

    # Ultralytics is chatty at INFO and prints a banner per image. We keep our
    # own output readable by raising its floor to WARNING; set MD_LOG_LEVEL to
    # DEBUG when you actually want its internals.
    if config.LOG_LEVEL != "DEBUG":
        logging.getLogger("ultralytics").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name, conventionally the caller's ``__name__``.

    Returns:
        A :class:`logging.Logger` writing to stdout at the configured level.
    """
    _configure_root_once()
    return logging.getLogger(name)
