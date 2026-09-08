"""Project-wide exception hierarchy.

Exceptions live in their own module, deliberately free of imports, so that any
module can raise a project error without taking on a dependency it does not
otherwise need. Before this existed, ``utils.dataset`` imported from
``utils.resources`` purely to reach the base class — a coupling that would have
propagated to every future module.

Having a single base class also gives callers a meaningful choice::

    except ModelDoctorError:   # something we anticipated and can explain
    except Exception:          # a genuine bug — should not be silenced
"""

from __future__ import annotations


class ModelDoctorError(Exception):
    """Base class for every error this project raises deliberately."""


class ResourceNotFoundError(ModelDoctorError):
    """A required file or directory the user must supply is not present."""


class ModelLoadError(ModelDoctorError):
    """The model file exists but could not be loaded as a YOLO model."""


class DatasetConfigError(ModelDoctorError):
    """``data.yaml`` is absent, unparseable, or missing required keys."""


class ExplainabilityError(ModelDoctorError):
    """A heatmap could not be produced."""
