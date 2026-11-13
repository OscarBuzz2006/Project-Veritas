"""Publication-grade SVG chart generators and static interactive lab notebook builder."""

from __future__ import annotations

from apart_epistemics.report.figures import generate_figures
from apart_epistemics.report.site import build_static_site

__all__ = [
    "build_static_site",
    "generate_figures",
]
