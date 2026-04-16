import json
import random
import time
from backendAllocater import allocate_resources

INTERVAL_SECONDS = 1
TOTAL_STEPS = 20
NUM_AGENTS = 5

OUTPUT_FILE = "output.json"


def generate_request_tuple():
    return tuple(random.randint(0, 16) for _ in range(NUM_AGENTS))


def run():
    for t in range(TOTAL_STEPS + 1):

        request_tuple = generate_request_tuple()

        print(f"\nt{t}")
        print("Request Tuple:")
        print(request_tuple)
        
        allocation_tuple = allocate_resources(request_tuple)
        
        print("Allocation Tuple:")
        print(allocation_tuple)

        # ✅ ADD THIS BLOCK (writes to frontend file)
        data = {
            "round": t,
            "request": list(request_tuple),      # convert tuple → list (JSON safe)
            "allocation": list(allocation_tuple)
        }

        with open(OUTPUT_FILE, "w") as f:
            json.dump(data, f)

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run()