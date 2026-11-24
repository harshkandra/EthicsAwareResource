import json
import time
from pathlib import Path

from agent_system import create_simulation, run_round

INTERVAL_SECONDS = 1
TOTAL_STEPS = 20
OUTPUT_FILE = Path(__file__).with_name("output.json")


def run():
    state = create_simulation()
    for round_number in range(1, TOTAL_STEPS + 1):
        result = run_round(state, round_number)
        print(json.dumps(result, indent=2))
        OUTPUT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if round_number < TOTAL_STEPS:
            time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run()