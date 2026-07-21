"""CI smoke test for the clamscan backend (real binary, not mocked).

The main pytest suite deliberately mocks every AV subprocess call, so
it stays fast and doesn't depend on ClamAV being installed on whatever
machine runs it — see tests/test_malware_scanner.py. This script is
the complement: it runs against a REAL `clamscan` binary + real virus
definitions to prove the clamscan backend actually works end-to-end on
the CI runner's OS, independent of config/malware_scan.yaml (which
ships with scanning disabled by default).

Usage (after `apt-get install -y clamav && freshclam`):
    python scripts/verify_clamscan.py

Exit code 0 = clean file scored clean AND EICAR scored infected.
Exit code 1 = anything else (binary missing, wrong verdict, exception).
"""
from __future__ import annotations

import shutil
import sys

from app.services.malware_scanner import MalwareScannerService, ScanVerdict

EICAR = rb"X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"


def main() -> int:
    binary = shutil.which("clamscan")
    if not binary:
        print("FAIL: clamscan not found on PATH — was `apt-get install clamav` run first?")
        return 1
    print(f"clamscan binary: {binary}")

    svc = MalwareScannerService(enabled=True, backend="clamscan", timeout_sec=120)

    clean = svc.scan_bytes(b"just an ordinary plain-text file, nothing malicious here", "clean.txt")
    print(f"clean.txt  -> verdict={clean.verdict.value} backend={clean.backend} detail={clean.detail}")

    # Calls the clamscan subprocess directly (bypassing the app's own
    # EICAR heuristic, which would short-circuit before the binary is
    # ever invoked) — this specifically proves the AV BINARY detects
    # it, not just our pre-check.
    infected = svc._scan_clamscan(EICAR, "eicar.com")
    print(f"eicar.com  -> verdict={infected.verdict.value} backend={infected.backend} detail={infected.detail}")

    ok = clean.verdict == ScanVerdict.CLEAN and infected.verdict == ScanVerdict.INFECTED
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
