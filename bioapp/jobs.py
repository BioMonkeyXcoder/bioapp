"""Background computation: a small pool of persistent spawned worker processes.

The GUI thread owns the pool. Each worker has its own Pipe; a QTimer polls the pipes while any
task is running, so no thread and no Qt object ever lives in a worker. A task is
``(task_id, target, payload)`` where target is ``"analysis"`` (payload = (kind, table,
dataset_ids, params), run through ``bioapp.analysis.run``) or ``"module:function"`` (called with the payload).
Cancelling a running task terminates its worker; the next dispatch starts a fresh one.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import traceback

from PySide6.QtCore import QObject, QTimer, Signal

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _run_analysis(payload):
    from bioapp import analysis
    kind, table, dataset_ids, params = payload
    return analysis.run(kind, table, dataset_ids, params)


def _worker_main(conn):
    try:  # warm the imports so the first real task does not pay for them
        from bioapp import analysis
        analysis.all_kinds()
    except Exception:
        pass
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        task_id, target, payload = msg
        try:
            if target == "analysis":
                fn = _run_analysis
            else:
                mod, _, name = target.partition(":")
                fn = getattr(importlib.import_module(mod), name)
            reply = (task_id, True, fn(payload))
        except BaseException:
            reply = (task_id, False, traceback.format_exc(limit=8))
        try:
            conn.send(reply)
        except BaseException:
            conn.send((task_id, False, "result could not be sent back:\n" + traceback.format_exc(limit=4)))


class _Worker:
    def __init__(self, ctx):
        saved = {k: os.environ.get(k) for k in _THREAD_VARS}
        os.environ.update({k: "1" for k in _THREAD_VARS})
        try:
            self.conn, child = ctx.Pipe()
            self.proc = ctx.Process(target=_worker_main, args=(child,), daemon=True)
            self.proc.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        child.close()
        self.task_id: str | None = None

    def kill(self):
        try:
            self.proc.terminate()
            self.proc.join(3)
        except Exception:
            pass
        self.conn.close()


class JobPool(QObject):
    """done(task_id, ok, result_or_error_text); state(running, queued) on every change."""
    done = Signal(str, bool, object)
    state = Signal(int, int)

    def __init__(self, n_workers: int = 2, parent=None):
        super().__init__(parent)
        self.n_workers = n_workers
        self._ctx = mp.get_context("spawn")
        self._workers: list[_Worker] = []
        self._queue: list[tuple[str, str, object]] = []
        self._timer = QTimer(self)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._poll)

    def warm(self):
        """Start one idle worker so the first task does not wait for process start-up."""
        if not self._workers:
            self._workers.append(_Worker(self._ctx))

    def submit(self, task_id: str, target: str, payload) -> None:
        self._queue.append((task_id, target, payload))
        self._dispatch()

    def cancel(self, task_id: str) -> bool:
        """Remove queued tasks and terminate the worker running ``task_id``. True if anything was cancelled."""
        n = len(self._queue)
        self._queue = [q for q in self._queue if q[0] != task_id]
        hit = n != len(self._queue)
        for w in list(self._workers):
            if w.task_id == task_id:
                w.kill()
                self._workers.remove(w)
                hit = True
        self._dispatch()
        return hit

    def cancel_all(self) -> None:
        self._queue.clear()
        for w in self._workers:
            if w.task_id is not None:
                w.kill()
        self._workers = [w for w in self._workers if w.task_id is None]
        self._dispatch()

    def shutdown(self) -> None:
        self._timer.stop()
        self._queue.clear()
        for w in self._workers:
            try:
                w.conn.send(None)
            except Exception:
                pass
            w.kill()
        self._workers.clear()

    def counts(self) -> tuple[int, int]:
        return sum(w.task_id is not None for w in self._workers), len(self._queue)

    def _dispatch(self) -> None:
        while self._queue:
            w = next((w for w in self._workers if w.task_id is None and w.proc.is_alive()), None)
            if w is None:
                self._workers = [x for x in self._workers if x.task_id is not None or x.proc.is_alive()]
                if len(self._workers) >= self.n_workers:
                    break
                w = _Worker(self._ctx)
                self._workers.append(w)
            task_id, target, payload = self._queue.pop(0)
            w.task_id = task_id
            try:
                w.conn.send((task_id, target, payload))
            except Exception as exc:  # unpicklable payload or broken pipe
                w.task_id = None
                self.done.emit(task_id, False, f"task could not be sent to the worker: {exc}")
        if any(w.task_id is not None for w in self._workers):
            self._timer.start()
        else:
            self._timer.stop()
        self.state.emit(*self.counts())

    def _poll(self) -> None:
        finished = []
        for w in list(self._workers):
            if w.task_id is None:
                continue
            try:
                if w.conn.poll():
                    tid, ok, res = w.conn.recv()
                    w.task_id = None
                    finished.append((tid, ok, res))
                elif not w.proc.is_alive():
                    finished.append((w.task_id, False, "worker process terminated unexpectedly"))
                    w.kill()
                    self._workers.remove(w)
            except (EOFError, OSError):
                finished.append((w.task_id, False, "worker process terminated unexpectedly"))
                w.kill()
                self._workers.remove(w)
        self._dispatch()  # free workers pick up queued tasks before listeners run
        for f in finished:
            self.done.emit(*f)
