import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict
from langgraph.graph import END, START, StateGraph


RESOURCE_KEYS = ("cpu", "ram_gb", "gpu", "network_mbps", "storage_iops")
CONFIG_PATH = Path(__file__).with_name("agentDetails.json")


@dataclass(frozen=True)
class AgentSpec:
    id: int
    name: str
    objective: str
    requirements: dict[str, int]
    minimums: dict[str, int]
    weights: dict[str, float]
    strategy: str
    base_priority: float
    workload_pattern: tuple[float, ...]

@dataclass
class AgentMemory:
    waiting_rounds: int = 0
    fulfilled_rounds: int = 0
    last_fulfillment: float = 1.0
    last_feedback: str = "Ready to negotiate."


@dataclass
class AgentProposal: 
    request: dict[str, int]
    bid: float
    waiting_rounds: int


@dataclass
class ResourceAgent:
    spec: AgentSpec
    memory: AgentMemory = field(default_factory=AgentMemory)

    def propose(self, round_number: int) -> AgentProposal:
        factor = self.spec.workload_pattern[(round_number - 1) % len(self.spec.workload_pattern)]
        if self.memory.waiting_rounds >= 2:
            factor = 1.0

        request = {
            resource: min(
                self.spec.requirements[resource],
                max(self.spec.minimums[resource], math.ceil(self.spec.requirements[resource] * factor)),
            )
            for resource in RESOURCE_KEYS
        }
        bid = (
            self.spec.base_priority * (1 + min(self.memory.waiting_rounds * 0.15, 0.75))
            + (1 - self.memory.last_fulfillment) * 0.25
        )
        return AgentProposal(request, bid, self.memory.waiting_rounds)

    def review(self, proposal: AgentProposal, allocation: dict[str, int]) -> float:
        fulfillment = sum(
            allocation[resource] / proposal.request[resource]
            for resource in RESOURCE_KEYS
            if proposal.request[resource] > 0
        ) / len(RESOURCE_KEYS)
        fulfillment = min(fulfillment, 1.0)
        fully_satisfied = all(
            allocation[resource] >= proposal.request[resource] for resource in RESOURCE_KEYS
        )

        self.memory.last_fulfillment = fulfillment
        self.memory.waiting_rounds = 0 if fully_satisfied else self.memory.waiting_rounds + 1
        if fully_satisfied:
            self.memory.fulfilled_rounds += 1
            self.memory.last_feedback = "Accepted: all requested resources were granted."
        else:
            preferred = max(self.spec.weights, key=self.spec.weights.get)
            self.memory.last_feedback = (
                f"Counter-proposal: prioritize {preferred.replace('_', ' ')} next round."
            )
        return fulfillment


@dataclass
class SimulationState:
    capacities: dict[str, int]
    agents: list[ResourceAgent]


class RoundGraphState(TypedDict):
    simulation: SimulationState
    round_number: int
    proposals: list[AgentProposal]
    allocations: list[dict[str, int]]
    result: dict[str, Any]


def _read_vector(value: Any, label: str, *, allow_zero: bool = True) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a resource object.")
    vector: dict[str, int] = {}
    for resource in RESOURCE_KEYS:
        amount = value.get(resource)
        if type(amount) is not int or amount < (0 if allow_zero else 1):
            raise ValueError(f"{label}.{resource} must be a valid integer.")
        vector[resource] = amount
    return vector


def load_configuration(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        configuration = json.load(file)
    if not isinstance(configuration, dict):
        raise ValueError("The simulation configuration must be a JSON object.")

    capacities = _read_vector(configuration.get("system_resources"), "system_resources")
    agent_configs = configuration.get("agents")
    if not isinstance(agent_configs, list) or not agent_configs:
        raise ValueError("At least one agent must be configured.")

    specs: list[AgentSpec] = []
    seen_ids: set[int] = set()
    for item in agent_configs:
        if not isinstance(item, dict):
            raise ValueError("Each agent configuration must be an object.")
        agent_id = item.get("id")
        name = item.get("name")
        if type(agent_id) is not int or not isinstance(name, str) or not name:
            raise ValueError("Every agent must have an integer id and a name.")
        if agent_id in seen_ids:
            raise ValueError(f"Duplicate agent id: {agent_id}.")
        seen_ids.add(agent_id)

        requirements = _read_vector(item.get("requirements"), f"{name}.requirements", allow_zero=False)
        minimums = _read_vector(item.get("minimums"), f"{name}.minimums")
        weights_raw = item.get("weights")
        if not isinstance(weights_raw, dict):
            raise ValueError(f"{name}.weights must be a resource object.")
        weights: dict[str, float] = {}
        for resource in RESOURCE_KEYS:
            weight = weights_raw.get(resource)
            if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
                raise ValueError(f"{name}.weights.{resource} must be positive.")
            weights[resource] = float(weight)
        for resource in RESOURCE_KEYS:
            if minimums[resource] > requirements[resource]:
                raise ValueError(f"{name} minimum {resource} exceeds its requirement.")

        pattern_raw = item.get("workload_pattern")
        if (
            not isinstance(pattern_raw, list)
            or not pattern_raw
            or any(
                not isinstance(factor, (int, float))
                or isinstance(factor, bool)
                or factor <= 0
                or factor > 1
                for factor in pattern_raw
            )
        ):
            raise ValueError(f"{name}.workload_pattern must contain factors in (0, 1].")
        base_priority = item.get("base_priority")
        if (
            not isinstance(base_priority, (int, float))
            or isinstance(base_priority, bool)
            or base_priority <= 0
        ):
            raise ValueError(f"{name}.base_priority must be positive.")

        specs.append(
            AgentSpec(
                id=agent_id,
                name=name,
                objective=str(item.get("objective", "")),
                requirements=requirements,
                minimums=minimums,
                weights=weights,
                strategy=str(item.get("strategy", "")),
                base_priority=float(base_priority),
                workload_pattern=tuple(float(factor) for factor in pattern_raw),
            )
        )

    for resource in RESOURCE_KEYS:
        minimum_total = sum(spec.minimums[resource] for spec in specs)
        if minimum_total > capacities[resource]:
            raise ValueError(
                f"Protected minimums for {resource} ({minimum_total}) exceed "
                f"capacity ({capacities[resource]})."
            )
    configuration["system_resources"] = capacities
    return configuration


def create_simulation(path: Path = CONFIG_PATH) -> SimulationState:
    configuration = load_configuration(path)
    agents = [
        ResourceAgent(
            AgentSpec(
                id=item["id"],
                name=item["name"],
                objective=str(item.get("objective", "")),
                requirements=item["requirements"],
                minimums=item["minimums"],
                weights={
                    resource: float(item["weights"][resource]) for resource in RESOURCE_KEYS
                },
                strategy=str(item.get("strategy", "")),
                base_priority=float(item["base_priority"]),
                workload_pattern=tuple(float(value) for value in item["workload_pattern"]),
            )
        )
        for item in configuration["agents"]
    ]
    return SimulationState(configuration["system_resources"], agents)


def _allocate(
    state: SimulationState, proposals: list[AgentProposal]
) -> list[dict[str, int]]:
    allocations = [
        {resource: agent.spec.minimums[resource] for resource in RESOURCE_KEYS}
        for agent in state.agents
    ]

    for resource in RESOURCE_KEYS:
        remaining = state.capacities[resource] - sum(
            allocation[resource] for allocation in allocations
        )
        while remaining > 0:
            eligible = [
                index
                for index, proposal in enumerate(proposals)
                if allocations[index][resource] < proposal.request[resource]
            ]
            if not eligible:
                break

            def marginal_utility(index: int) -> tuple[float, int]:
                agent = state.agents[index]
                proposal = proposals[index]
                discretionary = max(
                    proposal.request[resource] - agent.spec.minimums[resource], 1
                )
                extra_allocated = allocations[index][resource] - agent.spec.minimums[resource]
                fairness_boost = 1 + min(proposal.waiting_rounds, 5) * 0.1
                score = (
                    proposal.bid
                    * agent.spec.weights[resource]
                    * fairness_boost
                    / (1 + extra_allocated / discretionary)
                )
                return score, -agent.spec.id

            winner = max(eligible, key=marginal_utility)
            allocations[winner][resource] += 1
            remaining -= 1
    return allocations


def _propose_node(state: RoundGraphState) -> dict[str, Any]:
    proposals = [
        agent.propose(state["round_number"]) for agent in state["simulation"].agents
    ]
    return {"proposals": proposals}


def _allocate_node(state: RoundGraphState) -> dict[str, Any]:
    allocations = _allocate(state["simulation"], state["proposals"])
    return {"allocations": allocations}


def _review_node(state: RoundGraphState) -> dict[str, Any]:
    simulation = state["simulation"]
    proposals = state["proposals"]
    allocations = state["allocations"]
    agent_results: list[dict[str, Any]] = []
    fulfillment_scores: list[float] = []

    for agent, proposal, allocation in zip(simulation.agents, proposals, allocations):
        fulfillment = agent.review(proposal, allocation)
        fulfillment_scores.append(fulfillment)
        reasons = [
            "Protected minimums were allocated before discretionary resources.",
            f"Strategy: {agent.spec.strategy}",
        ]
        unmet = [
            resource.replace("_", " ")
            for resource in RESOURCE_KEYS
            if allocation[resource] < proposal.request[resource]
        ]
        if unmet:
            reasons.append(f"Negotiation left unmet demand for: {', '.join(unmet)}.")
        else:
            reasons.append("All requested resources were allocated.")

        agent_results.append(
            {
                "id": agent.spec.id,
                "name": agent.spec.name,
                "objective": agent.spec.objective,
                "strategy": agent.spec.strategy,
                "requested": proposal.request,
                "allocated": allocation,
                "minimum": agent.spec.minimums,
                "waiting": agent.memory.waiting_rounds,
                "fulfilled": agent.memory.fulfilled_rounds,
                "bid": round(proposal.bid, 3),
                "fulfillment": round(fulfillment, 3),
                "negotiation": agent.memory.last_feedback,
                "reasons": reasons,
            }
        )

    used = {
        resource: sum(allocation[resource] for allocation in allocations)
        for resource in RESOURCE_KEYS
    }
    remaining = {
        resource: simulation.capacities[resource] - used[resource]
        for resource in RESOURCE_KEYS
    }
    utilization = sum(
        used[resource] / simulation.capacities[resource]
        for resource in RESOURCE_KEYS
        if simulation.capacities[resource] > 0
    ) / sum(1 for resource in RESOURCE_KEYS if simulation.capacities[resource] > 0)
    fairness = (
        sum(fulfillment_scores) ** 2
        / (len(fulfillment_scores) * sum(score**2 for score in fulfillment_scores))
        if any(fulfillment_scores)
        else 1.0
    )

    return {
        "result": {
            "type": "round",
            "round": state["round_number"],
            "available_resources": simulation.capacities,
            "used_resources": used,
            "remaining_resources": remaining,
            "utilization": round(utilization, 3),
            "fairness_index": round(fairness, 3),
            "agents": agent_results,
        }
    }


def _build_round_graph():
    graph = StateGraph(RoundGraphState)
    graph.add_node("propose", _propose_node)
    graph.add_node("allocate", _allocate_node)
    graph.add_node("review", _review_node)
    graph.add_edge(START, "propose")
    graph.add_edge("propose", "allocate")
    graph.add_edge("allocate", "review")
    graph.add_edge("review", END)
    return graph.compile()


ROUND_GRAPH = _build_round_graph()


def run_round(state: SimulationState, round_number: int) -> dict[str, Any]:
    if type(round_number) is not int or round_number < 1:
        raise ValueError("round_number must be a positive integer.")
    result = ROUND_GRAPH.invoke(
        {
            "simulation": state,
            "round_number": round_number,
            "proposals": [],
            "allocations": [],
            "result": {},
        }
    )
    return result["result"]
