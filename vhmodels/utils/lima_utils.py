"""Utilities for running Apptainer through Lima on macOS."""

from contextlib import contextmanager

import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path


LIMA_INSTANCE = "vhmodels-apptainer"


def uses_lima():
    """Return whether Apptainer needs a Linux VM on this host."""
    return platform.system() == "Darwin"


def is_lima_shared_path(path):
    """Return whether ``path`` is in the macOS home mounted by Lima."""
    try:
        Path(path).expanduser().resolve().relative_to(Path.home().resolve())
    except ValueError:
        return False
    return True


def _parse_lima_instances(output):
    """Parse Lima's JSON-lines output (and older single JSON values)."""
    output = output.strip()
    if not output:
        return []
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return [json.loads(line) for line in output.splitlines() if line.strip()]
    return parsed if isinstance(parsed, list) else [parsed]


def _is_apple_silicon():
    """Detect the physical Mac architecture, including Python under Rosetta."""
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return True
    if machine not in {"x86_64", "amd64"}:
        return False
    for key in ("sysctl.proc_translated", "hw.optional.arm64"):
        try:
            result = subprocess.run(
                ["sysctl", "-in", key], capture_output=True, text=True
            )
        except FileNotFoundError:
            return False
        if result.returncode == 0 and result.stdout.strip() == "1":
            return True
    return False


@contextmanager
def _lima_instance_lock():
    """Serialize first VM creation across local processes."""
    import fcntl

    lock_path = Path(tempfile.gettempdir()) / f"vhmodels-lima-{os.getuid()}.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def _ensure_lima_instance():
    """Create or start the reusable macOS Apptainer VM."""

    try:
        result = subprocess.run(
            ["limactl", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )
        instances = _parse_lima_instances(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise RuntimeError("Failed to inspect Lima instances.") from error

    instance = next(
        (item for item in instances if item.get("name") == LIMA_INSTANCE), None
    )
    if instance is None:
        if _is_apple_silicon():
            architecture = ["--arch=aarch64", "--rosetta"]
        elif platform.machine().lower() in {"x86_64", "amd64"}:
            architecture = ["--arch=x86_64"]
        else:
            raise RuntimeError(
                f"Unsupported macOS architecture '{platform.machine().lower()}'."
            )
        command = [
            "limactl",
            "start",
            "--tty=false",
            f"--name={LIMA_INSTANCE}",
            "--vm-type=vz",
            *architecture,
            "--mount-writable",
            "template:apptainer",
        ]
    elif str(instance.get("status", "")).lower() != "running":
        command = ["limactl", "start", "--tty=false", LIMA_INSTANCE]
    else:
        return

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        hint = (
            " If Rosetta setup fails, run "
            "'softwareupdate --install-rosetta' and try again."
            if _is_apple_silicon()
            else ""
        )
        raise RuntimeError(f"Failed to prepare the Lima Apptainer VM.{hint}") from error


def ensure_lima_instance():
    """Prepare the Lima runtime once, without racing concurrent model calls."""
    if shutil.which("limactl") is None:
        raise RuntimeError(
            "Lima is not installed or 'limactl' is not in PATH. "
            "On macOS, install it with 'brew install lima'."
        )
    with _lima_instance_lock():
        _ensure_lima_instance()


def lima_shell_command(command, workdir):
    """Wrap an argv command for non-interactive execution in the Lima VM."""
    return [
        "limactl",
        "shell",
        "--tty=false",
        "--preserve-env",
        "--workdir",
        str(Path(workdir).resolve()),
        LIMA_INSTANCE,
        *command,
    ]
