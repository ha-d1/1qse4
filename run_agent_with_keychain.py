"""Launch the bounded agent using a SoCLaaS key stored in macOS Keychain."""
from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path

KEYCHAIN_SERVICE = "TikTok-TechJam-SoCLaaS"
DEFAULT_BASE_URL = "https://soclaas-api.comp.nus.edu.sg/v1"


def load_soclaas_key(account: str | None = None) -> str:
    account = account or getpass.getuser()
    result = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-a",
            account,
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    key = result.stdout.strip()
    if result.returncode != 0 or not key:
        raise RuntimeError(
            "SoCLaaS key not found in macOS Keychain. Run the one-time setup command "
            "from SOCLAAS_SETUP.md."
        )
    return key


def find_data_dir(root: Path, override: str | None = None) -> Path:
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    configured = os.environ.get("TECHJAM_DATA_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            root / "KuaiRand-Pure" / "data",
            Path.home()
            / "Documents"
            / "TikTok TechJam"
            / "1qse4-main"
            / "KuaiRand-Pure"
            / "data",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    raise RuntimeError("KuaiRand-Pure data directory was not found; pass --data-dir once.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--data-dir")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = find_data_dir(root, args.data_dir)
    key = load_soclaas_key()
    environment = os.environ.copy()
    environment.update(
        {
            "SOCLAAS_API_KEY": key,
            "SOCLAAS_BASE_URL": environment.get("SOCLAAS_BASE_URL", DEFAULT_BASE_URL),
            "SOCLAAS_MODEL": environment.get("SOCLAAS_MODEL", "qwen3-coder-next"),
        }
    )
    command = [
        sys.executable,
        str(root / "agent.py"),
        "--max-iterations",
        str(args.max_iterations),
        "--data-dir",
        str(data_dir),
    ]
    try:
        completed = subprocess.run(command, cwd=root, env=environment, check=False)
    finally:
        environment.pop("SOCLAAS_API_KEY", None)
        key = ""
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
