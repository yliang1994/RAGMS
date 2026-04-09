"""Read-only evaluation services and future runner interfaces."""

from typing import Any

from .dataset_loader import DatasetLoader, DatasetLoaderError
from .report_service import ReportService

__all__ = [
    "CompositeEvaluator",
    "DatasetLoader",
    "DatasetLoaderError",
    "EvalRunner",
    "ReportService",
    "build_evaluator_stack",
    "resolve_evaluator_backend_set",
]


def __getattr__(name: str) -> Any:
    if name in {"EvalRunner", "CompositeEvaluator", "build_evaluator_stack", "resolve_evaluator_backend_set"}:
        from .runner import CompositeEvaluator, EvalRunner, build_evaluator_stack, resolve_evaluator_backend_set

        exports = {
            "EvalRunner": EvalRunner,
            "CompositeEvaluator": CompositeEvaluator,
            "build_evaluator_stack": build_evaluator_stack,
            "resolve_evaluator_backend_set": resolve_evaluator_backend_set,
        }
        return exports[name]
    raise AttributeError(name)
