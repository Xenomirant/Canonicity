from __future__ import annotations

import argparse
import pathlib
import subprocess


REPOSITORY = "https://github.com/SheltonLiu-N/AutoDAN.git"
COMMIT = "34062e964185693e81a6775b4f0d00bfd7507612"


def output(*args: str, cwd: pathlib.Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the pinned AutoDAN reference assets")
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    destination = args.root.resolve() / "third_party" / "AutoDAN"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        subprocess.run(["git", "clone", REPOSITORY, str(destination)], check=True)
    if not (destination / ".git").exists():
        raise RuntimeError(f"Refusing to modify non-git directory: {destination}")
    subprocess.run(["git", "fetch", "origin", COMMIT], cwd=destination, check=True)
    subprocess.run(["git", "checkout", "--detach", COMMIT], cwd=destination, check=True)
    actual = output("git", "rev-parse", "HEAD", cwd=destination)
    if actual != COMMIT:
        raise RuntimeError(f"AutoDAN commit mismatch: expected {COMMIT}, got {actual}")
    asset = destination / "assets" / "prompt_group.pth"
    if not asset.exists():
        raise FileNotFoundError(asset)
    print(f"AutoDAN ready at {destination} ({actual})")


if __name__ == "__main__":
    main()
