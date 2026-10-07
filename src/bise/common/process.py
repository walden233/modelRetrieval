"""Stream each experiment command to the terminal and its own log."""

import os
import subprocess
import sys
from pathlib import Path


def run_logged(command: list, log_path: Path) -> None:
    args = [str(value) for value in command]
    print("RUN " + " ".join(args), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   bufsize=0, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            while chunk := os.read(process.stdout.fileno(), 4096):
                handle.write(chunk)
                handle.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, args)
