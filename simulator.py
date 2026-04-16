import random
import time
from backendAllocater import allocate_resources

INTERVAL_SECONDS = 1
TOTAL_STEPS = 20
NUM_AGENTS = 5   # fixed to 5 elements


def generate_request_tuple():
    # generate tuple of 5 random values between 0–16
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

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run()