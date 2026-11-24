import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "ethical-resource-flow-main" / "code"


def start_backend() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(ROOT / "server.py")], cwd=ROOT)


def start_frontend() -> subprocess.Popen:
    npm = "npm.cmd" if os.name == "nt" else "npm"
    return subprocess.Popen([npm, "run", "dev"], cwd=FRONTEND)


def main():
    processes = []
    try:
        processes.append(start_backend())
        processes.append(start_frontend())
        print("Starting the agent simulation and dashboard...")
        time.sleep(3)
        webbrowser.open("http://localhost:5173")
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping the application...")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()


if __name__ == "__main__":
    main()