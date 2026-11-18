"""Canonical project-class adapter for emission estimators.

The mapping implements the single class-adapter contract recorded as INV-012
in ``coordination/INVARIANTS.md``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

PROJECT_CLASS_TO_EMISSION_VEHICLE_TYPE: Final = MappingProxyType(
    {
        0: 6,  # bus -> transit bus
        1: 1,  # car -> passenger car
        2: 13,  # heavy truck -> combination truck
        3: 2,  # light truck -> passenger truck
    }
)


def emission_vehicle_type_for_project_class(class_id: int) -> int:
    """Return the native emission-model vehicle type for a project class.

    Raises:
        ValueError: If ``class_id`` is not one of the four project classes.
    """

    try:
        return PROJECT_CLASS_TO_EMISSION_VEHICLE_TYPE[class_id]
    except KeyError as exc:
        raise ValueError(f"Invalid project class_id: {class_id}") from exc


__all__ = [
    "PROJECT_CLASS_TO_EMISSION_VEHICLE_TYPE",
    "emission_vehicle_type_for_project_class",
]
