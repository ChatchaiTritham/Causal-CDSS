"""Regenerate only the result files the KBS manuscript uses, then check them against
results/SHA256SUMS_paper. Usage: python scripts/run_paper.py [--check-only]"""
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVERS = ["unmeasured_confounding.py", "proximal_robustness.py", "proximal_policy_subgroup.py",
           "proximal_bootstrap.py", "sample_size.py", "proximal_policy_violation.py",
           "proximal_audit_extensions.py", "crossing_uncertainty.py", "review_round2.py",
           "audit_card.py", "make_tables.py"]


def main():
    if "--check-only" not in sys.argv:
        for d in DRIVERS:
            subprocess.run([sys.executable, str(REPO / "scripts" / d)], check=True, cwd=REPO)
    bad = 0
    for line in (REPO / "results" / "SHA256SUMS_paper").read_text().splitlines():
        h, name = line.split(maxsplit=1)
        got = hashlib.sha256((REPO / "results" / name.strip()).read_bytes()).hexdigest()
        if got != h:
            bad += 1
            print("MISMATCH", name)
    print("all paper files match" if not bad else f"{bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
