import random
import time
import json
from backendAllocater import allocate_resources

INTERVAL_SECONDS = 1
TOTAL_STEPS = 20


def load_agents():
    with open("agentDetails.json", "r") as f:
        data = json.load(f)
    return data["agents"]


def generate_request_tuple(agents):

    request = []

    for agent in agents:

        cpu = random.randint(agent["min_cpu"], agent["max_cpu"])
        net = random.randint(agent["min_network"], agent["max_network"])
        ram = random.randint(agent["min_ram"], agent["max_ram"])

        request.append({
            "name": agent["name"],
            "cpu": cpu,
            "net": net,
            "ram": ram
        })

    return tuple(request)

def run():

    agents = load_agents()

    for t in range(TOTAL_STEPS + 1):

        request_tuple = generate_request_tuple(agents)

        cpu, ram, net = allocate_resources(request_tuple)

        print(f"\nt{t}")
        print("Requests:")

        for r in request_tuple:
            print(r)

        print("\nAllocated:")
        print("CPU:", cpu)
        print("RAM:", ram)
        print("NET:", net)

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run()