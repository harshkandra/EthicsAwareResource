import json
import random

FILE_PATH = "agentDetails.json"

ETHICS_CONFIG = {
    "wait_weight": 1.0,
    "need_weight": 0.7,
    "fulfilled_penalty": 0.5,
    "priority_base": 1.0,
    "starvation_boost": 8.0,
    "starvation_threshold": 3,
    "randomness_max": 0.2,
    "min_priority": 1,
    "max_priority": 100,
}


def load_data():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def resolve_priority_conflicts(agents):
    while True:
        seen = {}
        conflict_found = False

        for agent in agents:
            p = agent["priority_new"]

            if p in seen:
                current_balance = agent.get("waiting", 0) - agent.get("fullfilled", 0)
                seen_balance = seen[p].get("waiting", 0) - seen[p].get("fullfilled", 0)

                if current_balance > seen_balance:
                    agent["priority_new"] += 0.01
                else:
                    seen[p]["priority_new"] += 0.01

                conflict_found = True
                break
            else:
                seen[p] = agent

        if not conflict_found:
            break


def compute_bid(agent):
    priority_score = max(agent.get("priority_old", 1), 1)
    demand = agent.get("demand", 0)
    waiting = agent.get("waiting", 0)
    fulfilled = agent.get("fullfilled", 0)
    starvation_count = agent.get("starvation_count", 0)

    fairness_balance = waiting - fulfilled
    starvation_bonus = (
        ETHICS_CONFIG["starvation_boost"]
        if starvation_count >= ETHICS_CONFIG["starvation_threshold"]
        else 0
    )

    need_component = demand * ETHICS_CONFIG["need_weight"]
    fairness_component = fairness_balance * ETHICS_CONFIG["wait_weight"]
    fulfilled_component = fulfilled * ETHICS_CONFIG["fulfilled_penalty"]

    agent["need_score"] = demand
    agent["fairness_score"] = fairness_balance
    agent["ethics_score"] = (
        priority_score
        + need_component
        + fairness_component
        - fulfilled_component
        + starvation_bonus
    )

    randomness = random.uniform(0, ETHICS_CONFIG["randomness_max"])
    return agent["ethics_score"] * ETHICS_CONFIG["priority_base"] + randomness


def allocate_resources(request_tuple):
    data = load_data()

    total_available = data["system_resources"]["avilabe"]
    agents = data["agents"]

    for i, agent in enumerate(agents):
        agent["demand"] = request_tuple[i]
        agent["allocated"] = 0
        agent.setdefault("waiting", 0)
        agent.setdefault("fullfilled", 0)
        agent.setdefault("priority_old", 1)
        agent.setdefault("priority_new", 0)
        agent.setdefault("starvation_count", 0)
        agent.setdefault("need_score", 0)
        agent.setdefault("fairness_score", 0)
        agent.setdefault("ethics_score", 0)

    for agent in agents:
        agent["bid"] = compute_bid(agent)

    agents_sorted = sorted(agents, key=lambda x: x["bid"], reverse=True)
    remaining = total_available

    for agent in agents_sorted:
        if agent["starvation_count"] >= ETHICS_CONFIG["starvation_threshold"]:
            demand = agent["demand"]
            if demand > 0 and demand <= remaining:
                agent["allocated"] = demand
                remaining -= demand

    for agent in agents_sorted:
        if agent["allocated"] > 0:
            continue

        demand = agent["demand"]
        if demand <= 0 or demand > remaining:
            continue

        agent["allocated"] = demand
        remaining -= demand
        if remaining == 0:
            break

    for agent in agents:
        if agent["allocated"] == 0:
            agent["waiting"] += 1
            agent["starvation_count"] += 1
        else:
            agent["starvation_count"] = 0

        if agent["allocated"] == agent["demand"] and agent["demand"] > 0:
            agent["fullfilled"] += 1

    for agent in agents:
        fairness_balance = agent["waiting"] - agent["fullfilled"]
        agent["fairness_score"] = fairness_balance
        agent["ethics_score"] = (
            agent["priority_old"]
            + fairness_balance * ETHICS_CONFIG["wait_weight"]
            + agent["demand"] * ETHICS_CONFIG["need_weight"]
            - agent["fullfilled"] * ETHICS_CONFIG["fulfilled_penalty"]
        )

        agent["priority_new"] = max(
            min(agent["ethics_score"], ETHICS_CONFIG["max_priority"]),
            ETHICS_CONFIG["min_priority"],
        )

    resolve_priority_conflicts(agents)

    for agent in agents:
        agent["priority_old"] = agent["priority_new"]
        agent["priority_new"] = 0

    agents.sort(key=lambda x: x["id"])
    save_data(data)

    allocation_tuple = tuple(agent["allocated"] for agent in agents)
    return allocation_tuple
