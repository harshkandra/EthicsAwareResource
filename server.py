import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from websockets import serve
from websockets.exceptions import ConnectionClosed

from agent_system import create_simulation, load_configuration, run_round

config_lock = threading.Lock()
config: dict[str, Any] = {"rounds": 5, "interval": 1.0, "running": False}
clients: set[Any] = set()


async def broadcast(payload: dict[str, Any]) -> None:
    message = json.dumps(payload)
    for websocket in list(clients):
        try:
            await websocket.send(message)
        except ConnectionClosed:
            clients.discard(websocket)


async def ws_handler(websocket: Any) -> None:
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)


def is_running() -> bool:
    with config_lock:
        return bool(config["running"])


async def simulation_loop() -> None:
    while True:
        with config_lock:
            run_config = config.copy() if config["running"] else None
        if run_config is None:
            await asyncio.sleep(0.25)
            continue

        try:
            state = create_simulation()
            for round_number in range(1, run_config["rounds"] + 1):
                if not is_running():
                    break
                await broadcast(run_round(state, round_number))
                if round_number < run_config["rounds"]:
                    await asyncio.sleep(run_config["interval"])
            await broadcast({"type": "simulation_complete"})
        except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
            print(f"Simulation failed: {error}")
            await broadcast({"type": "simulation_error", "error": str(error)})
        finally:
            with config_lock:
                config["running"] = False


class Handler(BaseHTTPRequestHandler):
    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/run":
            self._send_json(404, {"error": "Not found."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length < 1 or content_length > 4096:
                raise ValueError("Request body must be between 1 and 4096 bytes.")
            request = json.loads(self.rfile.read(content_length))
            if not isinstance(request, dict):
                raise ValueError("Request body must be a JSON object.")
            rounds = request.get("rounds", 5)
            interval = request.get("interval", 1)
            if type(rounds) is not int or not 1 <= rounds <= 50:
                raise ValueError("rounds must be an integer between 1 and 50.")
            if (
                not isinstance(interval, (int, float))
                or isinstance(interval, bool)
                or not 0.1 <= interval <= 20
            ):
                raise ValueError("interval must be between 0.1 and 20 seconds.")
            load_configuration()
        except (ValueError, json.JSONDecodeError, OSError) as error:
            self._send_json(400, {"error": str(error)})
            return

        with config_lock:
            if config["running"]:
                self._send_json(409, {"error": "A simulation is already running."})
                return
            config.update(rounds=rounds, interval=float(interval), running=True)
        self._send_json(202, {"status": "started"})

    def do_GET(self) -> None:
        if self.path != "/system-state":
            self._send_json(404, {"error": "Not found."})
            return
        try:
            data = load_configuration()
        except (OSError, json.JSONDecodeError, ValueError) as error:
            self._send_json(500, {"error": str(error)})
            return
        self._send_json(
            200,
            {
                "available_resources": data["system_resources"],
                "agents": data["agents"],
            },
        )


def start_http() -> None:
    server = HTTPServer(("localhost", 8001), Handler)
    print("HTTP server running at http://localhost:8001")
    server.serve_forever()


async def main() -> None:
    threading.Thread(target=start_http, daemon=True).start()
    async with serve(ws_handler, "localhost", 8765):
        print("WebSocket running at ws://localhost:8765")
        await simulation_loop()


if __name__ == "__main__":
    asyncio.run(main())