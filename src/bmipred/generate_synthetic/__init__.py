# src/bmipred/__init__.py

from .synthetic_data import (
    set_seed,
    generate_patients,
    generate_health_assessment,
    generate_diagnosis,
    generate_medication,
    generate_lab_results,
    generate_hospital_admission,
    generate_family_history,
)

__all__ = [
    "set_seed",
    "generate_patients",
    "generate_health_assessment",
    "generate_diagnosis",
    "generate_medication",
    "generate_lab_results",
    "generate_hospital_admission",
    "generate_family_history",
]
