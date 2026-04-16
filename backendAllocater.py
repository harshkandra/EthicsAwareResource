import json
import random

FILE_PATH = "agentDetails.json"


def load_data():
    with open(FILE_PATH, "r") as f:
        return json.load(f)


def save_data(data):
    with open(FILE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def resolve_priority_conflicts(agents):
    while True:
        seen = {}
        conflict_found = False

        for agent in agents:
            p = agent["priority_new"]

            if p in seen:
                # lower priority_old gets +1
                if agent["priority_old"] < seen[p]["priority_old"]:
                    agent["priority_new"] += 1
                else:
                    seen[p]["priority_new"] += 1

                conflict_found = True
                break
            else:
                seen[p] = agent

        if not conflict_found:
            break


def compute_bid(agent):
    # avoid division by zero
    effective_priority = max(agent["priority_old"], 1)

    base = 1 / effective_priority
    waiting_boost = agent["waiting"]
    fulfilled_penalty = agent["fullfilled"]

    randomness = random.uniform(0, 0.3)

    return base + waiting_boost - fulfilled_penalty + randomness


def allocate_resources(request_tuple):
    data = load_data()

    total_available = data["system_resources"]["avilabe"]
    agents = data["agents"]

    # attach demand + reset allocation
    for i, agent in enumerate(agents):
        agent["demand"] = request_tuple[i]
        agent["allocated"] = 0

    # ⚔️ STEP 1: compute bids
    for agent in agents:
        agent["bid"] = compute_bid(agent)

    # ⚔️ STEP 2: sort bidders by highest bid
    agents_sorted = sorted(agents, key=lambda x: x["bid"], reverse=True)

    remaining = total_available
    allocated_agents = set()

    # ⚔️ STEP 3: negotiation loop
    for agent in agents_sorted:

        demand = agent["demand"]

        # skip if demand > remaining (as per rule)
        if demand > remaining:
            continue

        # allocate full demand (all-or-nothing)
        agent["allocated"] = demand
        remaining -= demand

        allocated_agents.add(agent["id"])

        # stop if no resource left
        if remaining == 0:
            break

    # STEP 4: update waiting & fulfilled
    for agent in agents:
        if agent["allocated"] == 0:
            agent["waiting"] += 1

        if agent["allocated"] == agent["demand"] and agent["demand"] > 0:
            agent["fullfilled"] += 1

    # STEP 5: compute new priority
    for agent in agents:
        agent["priority_new"] = (
            agent["priority_old"]
            - agent["waiting"]
            + agent["fullfilled"]
        )

    # STEP 6: resolve conflicts
    resolve_priority_conflicts(agents)

    # STEP 7: update priorities (avoid ≤0)
    for agent in agents:
        agent["priority_old"] = max(agent["priority_new"], 1)
        agent["priority_new"] = 0

    # restore original order (by id)
    agents.sort(key=lambda x: x["id"])

    # save updated state
    save_data(data)

    # return allocation tuple
    allocation_tuple = tuple(agent["allocated"] for agent in agents)

    return allocation_tuple