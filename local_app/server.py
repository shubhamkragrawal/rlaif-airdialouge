"""Start the localhost conversation API with ``python -m local_app.server``."""

from __future__ import annotations

import argparse

from .api import SessionAPI, make_server
from .audit import AuditLog
from .controller import ConversationController


def _runtime_status_provider(manager):
    """Adapt RuntimeManager snapshots to the browser's secret-free status envelope."""
    def provide(adapter):
        runtime = manager.status(adapter)
        lifecycle = runtime.get("lifecycle_status", runtime.get("status"))
        if lifecycle not in {"loading", "ready", "available", "configured", "unavailable", "incompatible"}:
            lifecycle = "unavailable"
        provenance_snapshot = manager.provenance(adapter)
        provenance_status = provenance_snapshot.get("provenance_status")
        if provenance_status not in {"complete", "partial", "pending", "unavailable", "error"}:
            provenance_status = "unavailable"
        usable = runtime.get("mode") == "local_model" or lifecycle in {"loading", "ready", "available"}
        return {
            "adapter": adapter,
            "lifecycle_status": lifecycle,
            "mode": "local_model" if usable else "deterministic_fallback",
            "available": bool(runtime.get("available", lifecycle in {"ready", "available"})),
            "provenance_status": provenance_status,
        }
    return provide


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the local RLAIF conversation API")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (localhost by default)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", action="store_true",
                        help="lazily load the selected local Qwen/PEFT adapter")
    args = parser.parse_args(argv)
    manager = None
    if args.model:
        from .runtime import RuntimeManager
        manager = RuntimeManager()
    controller = ConversationController(
        audit=AuditLog(), model_loader=manager.get if manager else None
    )
    server = make_server(args.host, args.port, SessionAPI(
        controller, runtime_status_provider=_runtime_status_provider(manager) if manager else None
    ))
    print(f"RLAIF local API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local API")
    finally:
        server.server_close()
        if manager:
            manager.release()


if __name__ == "__main__":
    main()
