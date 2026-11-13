"""Record the reference digests for tests/test_regression.py.

Run this only when an environment version is added or bumped, never to make a failing
regression test pass:

    python tests/regression/record_reference.py [--only CASE ...]

It records with whichever ``gridhunt`` is importable, so put the version you want to record
first on PYTHONPATH.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import trajectories  # noqa: E402

import gridhunt  # noqa: E402

REFERENCE = Path(__file__).parent / "reference_trajectories.json.gz"

# IDs the environments had before they were renamed for the 1.0 release
_EARLIER_IDS = {"GridHunt-PigChase-v0": "GridHunt-PIG-v0"}


def _registered_id(env_id):
    if env_id in gridhunt._BENCHMARKS:
        return env_id
    return _EARLIER_IDS[env_id]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="Re-record only these cases and keep the rest")
    args = parser.parse_args()

    reference = {}
    if args.only and REFERENCE.exists():
        with gzip.open(REFERENCE, "rt") as f:
            reference = json.load(f)["cases"]
    names = args.only or list(trajectories.CASES)
    for name in names:
        case = trajectories.CASES[name]
        reference[name] = {}
        for seed in trajectories.SEEDS:
            env = trajectories.make_case_env(gridhunt, case, env_id=_registered_id(case["env_id"]))
            reference[name][str(seed)] = trajectories.run_case(env, seed)
            env.close()
        print(f"{name}: {len(reference[name][str(trajectories.SEEDS[0])])} digests per seed")

    payload = {
        "steps": trajectories.STEPS,
        "max_steps": trajectories.MAX_STEPS,
        "seeds": list(trajectories.SEEDS),
        "cases": {name: reference[name] for name in sorted(reference)},
    }
    # mtime=0 keeps the file byte-identical when nothing changed
    with gzip.GzipFile(REFERENCE, "wb", mtime=0) as f:
        f.write(json.dumps(payload, indent=0, sort_keys=True).encode())
    print(f"Wrote {REFERENCE}")


if __name__ == "__main__":
    main()
