from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

from mlxr.core.runtime import (
    CatalogConflictError,
    CatalogNotFoundError,
    CatalogValidationError,
    RuntimeCatalog,
)
from mlxr.core.schemas import (
    ModelInstallOperationRecord,
    SupportedModelDescriptor,
)

from .store import ModelInstallStore

_TERMINAL_PHASES = {"completed", "failed", "cancelled"}
_UNSET = object()


class ModelInstallConflictError(Exception):
    pass


class ModelInstallNotFoundError(Exception):
    pass


class ModelInstallManager:
    def __init__(self, catalog: RuntimeCatalog, store: ModelInstallStore) -> None:
        self.catalog = catalog
        self.store = store
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._active_by_model_id: dict[str, str] = {}
        self._worker: threading.Thread | None = None
        self._recover_persisted_operations()

    def enqueue(self, model_id: str) -> ModelInstallOperationRecord:
        supported_model = self._supported_model(model_id)

        with self._lock:
            existing_operation_id = self._active_by_model_id.get(model_id)
            if existing_operation_id is not None:
                existing_record = self.store.get(existing_operation_id)
                if existing_record is not None:
                    return existing_record

            existing_model = self.catalog.models.get(model_id)
            if existing_model is not None and existing_model.artifact is not None:
                result = self.catalog.install_supported_model(model_id)
                completed = ModelInstallOperationRecord(
                    operation_id=f"mdl_{uuid.uuid4().hex}",
                    model_id=model_id,
                    phase="completed",
                    supported_model=supported_model,
                    result=result,
                    created_at=_utcnow(),
                    updated_at=_utcnow(),
                    started_at=_utcnow(),
                    finished_at=_utcnow(),
                )
                return self.store.save(completed)

            record = ModelInstallOperationRecord(
                operation_id=f"mdl_{uuid.uuid4().hex}",
                model_id=model_id,
                phase="queued",
                supported_model=supported_model,
            )
            self.store.save(record)
            self._queue.append(record.operation_id)
            self._active_by_model_id[model_id] = record.operation_id
            self._ensure_worker_locked()
            return record

    def list_operations(self) -> list[ModelInstallOperationRecord]:
        return self.store.list_records()

    def get_operation(self, operation_id: str) -> ModelInstallOperationRecord:
        record = self.store.get(operation_id)
        if record is None:
            raise ModelInstallNotFoundError(
                f"Unknown model install operation '{operation_id}'"
            )
        return record

    def cancel(self, operation_id: str) -> ModelInstallOperationRecord:
        with self._lock:
            record = self.get_operation(operation_id)
            if record.phase != "queued":
                raise ModelInstallConflictError(
                    f"Install operation '{operation_id}' is already running"
                )
            self._queue = [item for item in self._queue if item != operation_id]
            self._active_by_model_id.pop(record.model_id, None)
            cancelled = record.model_copy(
                update={
                    "phase": "cancelled",
                    "updated_at": _utcnow(),
                    "finished_at": _utcnow(),
                }
            )
            return self.store.save(cancelled)

    def has_active_operation(self, model_id: str) -> bool:
        with self._lock:
            operation_id = self._active_by_model_id.get(model_id)
            if operation_id is None:
                return False
            record = self.store.get(operation_id)
            return record is not None and record.phase not in _TERMINAL_PHASES

    def _recover_persisted_operations(self) -> None:
        for record in self.store.list_records():
            if record.phase in _TERMINAL_PHASES:
                continue
            failed = record.model_copy(
                update={
                    "phase": "failed",
                    "error": "The runtime restarted before this install completed.",
                    "updated_at": _utcnow(),
                    "finished_at": _utcnow(),
                }
            )
            self.store.save(failed)

    def _ensure_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._run_loop,
            name="mlxr-model-install-worker",
            daemon=True,
        )
        self._worker.start()

    def _run_loop(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    self._worker = None
                    return
                operation_id = self._queue.pop(0)

            try:
                self._run_operation(operation_id)
            finally:
                with self._lock:
                    record = self.store.get(operation_id)
                    if record is not None and record.phase in _TERMINAL_PHASES:
                        self._active_by_model_id.pop(record.model_id, None)

    def _run_operation(self, operation_id: str) -> None:
        try:
            record = self.get_operation(operation_id)
            self._save(
                record,
                phase="resolving",
                started_at=record.started_at or _utcnow(),
                preview=self.catalog.preview_supported_model(record.model_id),
            )
            result = self.catalog.install_supported_model(
                self.get_operation(operation_id).model_id,
                phase_callback=lambda phase: self._save_phase(operation_id, phase),
            )
        except PermissionError as exc:
            self._save(
                self.get_operation(operation_id),
                phase="auth_required",
                error=str(exc),
                finished_at=_utcnow(),
            )
            return
        except (
            CatalogConflictError,
            CatalogNotFoundError,
            CatalogValidationError,
            FileNotFoundError,
            ValueError,
        ) as exc:
            self._save(
                self.get_operation(operation_id),
                phase="failed",
                error=str(exc),
                finished_at=_utcnow(),
            )
            return

        self._save(
            self.get_operation(operation_id),
            phase="completed",
            result=result,
            error=None,
            finished_at=_utcnow(),
        )

    def _save_phase(self, operation_id: str, phase: str) -> None:
        self._save(self.get_operation(operation_id), phase=phase)

    def _save(
        self,
        record: ModelInstallOperationRecord,
        *,
        phase: str | None = None,
        preview: object = _UNSET,
        result: object = _UNSET,
        error: str | object = _UNSET,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> ModelInstallOperationRecord:
        updated = record.model_copy(
            update={
                "phase": phase or record.phase,
                "preview": record.preview if preview is _UNSET else preview,
                "result": record.result if result is _UNSET else result,
                "error": record.error if error is _UNSET else error,
                "updated_at": _utcnow(),
                "started_at": (
                    started_at if started_at is not None else record.started_at
                ),
                "finished_at": (
                    finished_at if finished_at is not None else record.finished_at
                ),
            }
        )
        return self.store.save(updated)

    def _supported_model(self, model_id: str) -> SupportedModelDescriptor:
        for item in self.catalog.list_supported_models():
            if item.model_id == model_id:
                return item
        raise CatalogNotFoundError(f"Unknown supported model '{model_id}'")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
