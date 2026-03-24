import json


def load_agent_config():
    with open("agentDetails.json", "r") as f:
        return json.load(f)


def compute_scores(requests, agents):

    scores = []

    for i, agent in enumerate(agents):

        req = requests[i]

        priority = agent["priority"]

        # demand-based moral score
        demand = req["cpu"] + req["ram"] + req["net"]

        score = demand * (1 / priority)

        scores.append(score)

    return scores


def allocate_resource(requests, agents, total, resource):

    # Step 1: give minimum
    allocation = [
        agent[f"min_{resource}"] for agent in agents
    ]

    remaining = total - sum(allocation)

    if remaining <= 0:
        return allocation

    # Step 2: requested extras
    requested_extra = []

    for i, agent in enumerate(agents):

        req_key = resource if resource != "network" else "net"
        req = requests[i][req_key]
        min_v = agent[f"min_{resource}"]

        extra = max(0, req - min_v)
        requested_extra.append(extra)

    # Step 3: normalize extras to remaining
    total_extra = sum(requested_extra)

    if total_extra == 0:
        return allocation

    scaled_extra = [
        (x / total_extra) * remaining
        for x in requested_extra
    ]

    # Step 4: add but clamp to max
    for i, agent in enumerate(agents):

        max_v = agent[f"max_{resource}"]

        allocation[i] += scaled_extra[i]

        if allocation[i] > max_v:
            allocation[i] = max_v

    # Step 5: redistribute leftover
    while True:

        used = sum(allocation)
        leftover = total - used

        if leftover <= 0:
            break

        eligible = []

        for i, agent in enumerate(agents):
            if allocation[i] < agent[f"max_{resource}"]:
                eligible.append(i)

        if not eligible:
            break

        share = leftover / len(eligible)

        for i in eligible:

            max_v = agents[i][f"max_{resource}"]

            allocation[i] += share

            if allocation[i] > max_v:
                allocation[i] = max_v

    return allocation


def allocate_resources(request_tuple):

    config = load_agent_config()

    agents = config["agents"]
    system = config["system_resources"]

    cpu = allocate_resource(
        request_tuple,
        agents,
        system["total_cpu_cores"],
        "cpu"
    )

    ram = allocate_resource(
        request_tuple,
        agents,
        system["total_ram_gb"],
        "ram"
    )

    net = allocate_resource(
        request_tuple,
        agents,
        system["total_network_mbps"],
        "network"
    )

    return (
        tuple(round(x) for x in cpu),
        tuple(round(x) for x in ram),
        tuple(round(x) for x in net)
    )