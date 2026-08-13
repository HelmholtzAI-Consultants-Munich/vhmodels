"""Helpers for running and terminating subprocess trees."""

import os
import signal
import subprocess


def terminate_process_group(process):
    """Terminate and reap a subprocess together with all of its descendants."""
    try:
        process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        process.wait()
        return

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError(
        f"Could not terminate model worker process group {process_group}."
    )


def run_subprocess(command, payload, subprocess_env, timeout):
    """Run a bounded subprocess and return its captured stdout and stderr."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=subprocess_env,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=payload, timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process_group(process)
        stdout, stderr = process.communicate()
        raise RuntimeError(
            f"Model subprocess exceeded {timeout}s and was killed.\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        )
    except BaseException:
        terminate_process_group(process)
        process.communicate()
        raise
    if process.returncode != 0:
        raise RuntimeError(
            f"Model subprocess failed (exit {process.returncode}).\n"
            f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        )
    return stdout, stderr
