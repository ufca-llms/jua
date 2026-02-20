from jua.models import get_model
from jua.benchmarks import get_jua_tasks, get_dataset_tasks, get_all_tasks
from .run import run


def get_tasks(benchmark: str, **kwargs):
    if benchmark.lower() == "jua":
        return get_jua_tasks(**kwargs)
    return get_dataset_tasks(benchmark, registry_path=kwargs.get("dataset_registry"))


def get_all_benchmark_tasks(dataset_registry: str | None = None):
    return get_all_tasks(registry_path=dataset_registry)


__all__ = ["get_model", "get_tasks", "get_all_benchmark_tasks", "run"]
