import subprocess
import webbrowser
import time
import os

PORT = 8000

def start_simulator():
    return subprocess.Popen(["python", "simulator.py"])


def start_server():
    return subprocess.Popen(["python", "-m", "http.server", str(PORT)])


def open_browser():
    time.sleep(2)  # wait for server to start
    webbrowser.open(f"http://localhost:{PORT}")


def main():
    print("🚀 Starting system...")

    sim_process = start_simulator()
    server_process = start_server()

    open_browser()

    try:
        sim_process.wait()
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        sim_process.terminate()
        server_process.terminate()


if __name__ == "__main__":
    main()