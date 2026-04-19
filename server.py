import asyncio
import json
import random
from websockets import serve
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

from ethicalAllocater import allocate_resources, load_data

NUM_AGENTS = 5

# shared config
config = {
    "rounds": 20,
    "interval": 1,
    "running": False
}

clients = set()


def get_system_resource_limit():
    """Load system resource limit from agentDetails.json"""
    try:
        data = load_data()
        return data.get("system_resources", {}).get("avilabe", 16)
    except (FileNotFoundError, json.JSONDecodeError):
        return 16  # Fallback to default


def generate_request_tuple():
    limit = get_system_resource_limit()
    return tuple(random.randint(0, limit) for _ in range(NUM_AGENTS))


def explain_agent_reason(agent):
    reasons = []
    demand = agent.get("demand", 0)
    allocation = agent.get("allocated", 0)
    ethics_score = round(agent.get("ethics_score", 0), 2)
    waiting = agent.get("waiting", 0)
    fulfilled = agent.get("fullfilled", 0)
    priority = agent.get("priority_old", 1)

    reasons.append(f"Priority score: {priority}.")
    reasons.append(f"Ethical score: {ethics_score}.")

    if demand <= 0:
        reasons.append("No demand this round.")
    elif allocation == demand:
        reasons.append("Demand was fully satisfied.")
        reasons.append("The agent had a strong ethical bid based on need and waiting history.")
        if waiting > 0:
            reasons.append("Waiting history increased future fairness priority.")
    elif allocation == 0:
        reasons.append("No resources remained after higher-priority agents were allocated.")
        reasons.append("Current ethical bid was not enough this round.")
        if waiting > 0:
            reasons.append("Waiting will increase its chance in later rounds.")
    else:
        reasons.append("Allocated partially because remaining resources were limited.")

    if fulfilled > 0:
        reasons.append("Past fulfilment reduced priority slightly to preserve fairness.")

    return reasons


# ---------- WEBSOCKET ----------
async def ws_handler(websocket):
    clients.add(websocket)
    print("Client connected")

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        clients.remove(websocket)


# ---------- SIMULATION ----------
async def simulation_loop():
    t = 0

    while True:
        if config["running"]:
            request = generate_request_tuple()
            allocation = allocate_resources(request)

            data = load_data()
            agents = data["agents"]
            available_resources = data.get("system_resources", {}).get("avilabe", 0)

            payload = {
                "round": t,
                "request": list(request),
                "allocation": list(allocation),
                "available_resources": available_resources,
                "agents": [
                    {
                        "name": a["name"],
                        "waiting": a["waiting"],
                        "fullfilled": a["fullfilled"],
                        "bid": round(a.get("bid", 0), 3),
                        "demand": a["demand"],
                        "allocated": a["allocated"],
                        "priority": a.get("priority_old", 1),
                        "ethics_score": round(a.get("ethics_score", 0), 2),
                        "reasons": explain_agent_reason(a),
                    }
                    for a in agents
                ]
            }

            # send to all clients
            for ws in list(clients):
                try:
                    await ws.send(json.dumps(payload))
                except Exception as exc:
                    print("WebSocket send failed, removing client:", exc)
                    clients.discard(ws)

            t += 1

            if t >= config["rounds"]:
                config["running"] = False
                t = 0

            await asyncio.sleep(config["interval"])
        else:
            await asyncio.sleep(0.5)


# ---------- HTTP SERVER ----------
class Handler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def _parse_int(self, value, default):
        try:
            if value is None:
                return default
            return int(value)
        except (ValueError, TypeError):
            return default

    def do_POST(self):
        if self.path == "/run":
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)

            data = json.loads(body or "{}")

            config["rounds"] = self._parse_int(data.get("rounds"), 20)
            config["interval"] = self._parse_int(data.get("interval"), 1)
            config["running"] = True

            print("Simulation started", config)

            self.send_response(200)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(b"Started")

        else:
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()

    def do_GET(self):
        if self.path == "/system-state":
            data = load_data()
            available_resources = data.get("system_resources", {}).get("avilabe", 0)
            
            response = {
                "available_resources": available_resources
            }
            
            self.send_response(200)
            self._set_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()


def start_http():
    server = HTTPServer(("localhost", 8001), Handler)
    print("HTTP server running at http://localhost:8001")
    server.serve_forever()


# ---------- MAIN ----------
async def main():
    threading.Thread(target=start_http, daemon=True).start()

    async with serve(ws_handler, "localhost", 8765):
        print("WebSocket running at ws://localhost:8765")
        await simulation_loop()


asyncio.run(main())