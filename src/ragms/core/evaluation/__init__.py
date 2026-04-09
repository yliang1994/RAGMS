"""Read-only evaluation services and future runner interfaces."""

from .dataset_loader import DatasetLoader, DatasetLoaderError
from .report_service import ReportService
from .runner import EvalRunner, CompositeEvaluator, build_evaluator_stack, resolve_evaluator_backend_set

__all__ = [
    "CompositeEvaluator",
    "DatasetLoader",
    "DatasetLoaderError",
    "EvalRunner",
    "ReportService",
    "build_evaluator_stack",
    "resolve_evaluator_backend_set",
]
