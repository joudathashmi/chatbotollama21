"""Production-style local launcher.

Unlike run.py (dev server with auto-reload), this runs the app the way
it should run in a deployment: no reload, multiple worker processes,
bound to all interfaces, access logging on.

    python serve.py

Tunable via environment:
    MISA_HOST     (default 0.0.0.0)
    MISA_PORT     (default 8000)
    MISA_WORKERS  (default: min(4, cpu_count) )
"""

import os
import sys
import multiprocessing

import uvicorn

if __name__ == "__main__":
    host = os.getenv("MISA_HOST", "0.0.0.0")
    port = int(os.getenv("MISA_PORT", "8000"))
    default_workers = min(4, max(1, multiprocessing.cpu_count() // 2))
    workers = int(os.getenv("MISA_WORKERS", str(default_workers)))

    # Import after env is readable so config sees MISA_* vars.
    from app import config
    rl_err = config.require_redis_rate_limit_for_workers(workers)
    if rl_err:
        print(f"SECURITY: {rl_err}", file=sys.stderr)
        if config.IS_PRODUCTION:
            sys.exit(1)
        print(
            "Continuing in non-production mode — set MISA_RATE_LIMIT_BACKEND=redis "
            "before scaling workers.",
            file=sys.stderr,
        )

    print(f"Starting MISA Intelligence API — {host}:{port} · {workers} workers "
          f"(production mode, no reload)")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        access_log=True,
        log_level="info",
    )
