"""Model Doctor: why a vision model's predictions fail, measured from a run.

The package groups what used to sit at the top of the source tree — ``config``,
``app`` and ``utils`` — under one name. Nothing about the analysis changed; the
move exists so another application can install this alongside its own code
without the two projects' ``utils`` packages hiding one another.

Import the pieces directly::

    from model_doctor import config
    from model_doctor.app import storage, jobs

``model_doctor.app.api`` and ``model_doctor.app.control`` remain the standalone HTTP
entry points, and
every stage is still a module runnable with ``python -m``.
"""

__all__ = ["app", "config", "utils"]
