import unittest
from pathlib import Path

from agent_system import RESOURCE_KEYS, create_simulation, run_round


class AgentSystemTests(unittest.TestCase):
    def test_rounds_never_exceed_capacity_or_violate_minimums(self) -> None:
        state = create_simulation()

        for round_number in range(1, 9):
            result = run_round(state, round_number)
            for resource in RESOURCE_KEYS:
                self.assertLessEqual(
                    result["used_resources"][resource],
                    result["available_resources"][resource],
                )
                self.assertEqual(
                    result["used_resources"][resource]
                    + result["remaining_resources"][resource],
                    result["available_resources"][resource],
                )
            for agent in result["agents"]:
                for resource in RESOURCE_KEYS:
                    self.assertGreaterEqual(
                        agent["allocated"][resource], agent["minimum"][resource]
                    )
                    self.assertLessEqual(
                        agent["allocated"][resource], agent["requested"][resource]
                    )

    def test_agents_keep_independent_workload_patterns_and_feedback(self) -> None:
        state = create_simulation(Path(__file__).resolve().parents[1] / "agentDetails.json")
        first_round = run_round(state, 1)
        second_round = run_round(state, 2)

        self.assertEqual(
            [agent["name"] for agent in first_round["agents"]],
            ["Coding Agent", "ML Training Agent", "DevOps Agent"],
        )
        self.assertNotEqual(
            first_round["agents"][0]["requested"],
            first_round["agents"][1]["requested"],
        )
        self.assertTrue(
            any(
                agent["negotiation"].startswith("Counter-proposal")
                for agent in second_round["agents"]
            )
        )
        for before, after in zip(first_round["agents"], second_round["agents"]):
            if before["fulfillment"] < 1:
                self.assertGreater(after["bid"], before["bid"])
        self.assertTrue(0 <= second_round["fairness_index"] <= 1)
        self.assertTrue(0 <= second_round["utilization"] <= 1)

    def test_rejects_capacity_below_protected_minimums(self) -> None:
        with self.assertRaisesRegex(ValueError, "Protected minimums for network_mbps"):
            create_simulation_with_network_capacity(574)


def create_simulation_with_network_capacity(capacity: int):
    import json
    import tempfile

    source = Path(__file__).resolve().parents[1] / "agentDetails.json"
    configuration = json.loads(source.read_text(encoding="utf-8"))
    configuration["system_resources"]["network_mbps"] = capacity
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "configuration.json"
        path.write_text(json.dumps(configuration), encoding="utf-8")
        return create_simulation(path)


if __name__ == "__main__":
    unittest.main()
