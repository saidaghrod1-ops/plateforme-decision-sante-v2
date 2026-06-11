"""
Couche ORCHESTRATION — exécuteur de DAG de tâches.

Mini-moteur de workflow : chaque tâche déclare ses dépendances ; l'exécuteur les
ordonne topologiquement et propage un contexte partagé. En production, remplacer
par Airflow / Prefect / Celery sans changer les services (mêmes signatures).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from core.logging import get_logger

log = get_logger("orchestration")


@dataclass
class Task:
    name: str
    fn: Callable[[dict], dict]          # (context) -> mise à jour du contexte
    depends_on: list[str] = field(default_factory=list)


class Pipeline:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add(self, task: Task) -> "Pipeline":
        self._tasks[task.name] = task
        return self

    def _topological_order(self) -> list[str]:
        visited, order = set(), []

        def visit(name: str, stack: set[str]):
            if name in visited:
                return
            if name in stack:
                raise ValueError(f"Cycle détecté autour de {name}")
            stack.add(name)
            for dep in self._tasks[name].depends_on:
                visit(dep, stack)
            stack.discard(name)
            visited.add(name)
            order.append(name)

        for name in self._tasks:
            visit(name, set())
        return order

    def run(self, context: dict | None = None) -> dict:
        context = context or {}
        for name in self._topological_order():
            log.info("Exécution de la tâche : %s", name)
            update = self._tasks[name].fn(context) or {}
            context.update(update)
        return context
