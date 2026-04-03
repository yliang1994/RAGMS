"""Shared batching utilities for deterministic embedding-stage scheduling."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def optimize_embedding_batches(
    items: Sequence[InputT],
    process_batch: Callable[[list[InputT]], list[OutputT]],
    *,
    batch_size: int = 32,
    max_workers: int = 1,
    cache: dict[Hashable, OutputT] | None = None,
    cache_key_fn: Callable[[InputT], Hashable] | None = None,
) -> list[OutputT]:
    """Process inputs in stable batches with optional cache reuse and concurrency."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than zero")

    if not items:
        return []

    key_fn = cache_key_fn or (lambda item: item)  # type: ignore[return-value]
    shared_cache = cache if cache is not None else {}
    ordered_keys = [key_fn(item) for item in items]

    unique_items: list[InputT] = []
    unique_keys: list[Hashable] = []
    seen_keys: set[Hashable] = set(shared_cache)
    for item, key in zip(items, ordered_keys, strict=True):
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_items.append(item)
        unique_keys.append(key)

    batches = [
        (
            unique_keys[index : index + batch_size],
            unique_items[index : index + batch_size],
        )
        for index in range(0, len(unique_items), batch_size)
    ]

    def store_batch_results(batch_keys: list[Hashable], batch_results: list[OutputT]) -> None:
        if len(batch_results) != len(batch_keys):
            raise ValueError("Processed batch size does not match input batch size")
        for key, result in zip(batch_keys, batch_results, strict=True):
            shared_cache[key] = deepcopy(result)

    if batches:
        if max_workers == 1:
            for batch_keys, batch_items in batches:
                store_batch_results(batch_keys, process_batch(list(batch_items)))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_batch, list(batch_items)): list(batch_keys)
                    for batch_keys, batch_items in batches
                }
                for future in as_completed(futures):
                    store_batch_results(futures[future], future.result())

    return [deepcopy(shared_cache[key]) for key in ordered_keys]
