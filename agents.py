# agents.py
from mesa import Agent
import numpy as np

class ResourceAgent(Agent):
    def __init__(self, unique_id, model, need_level, moral_weight=0.5):
        super().__init__(unique_id, model)
        self.need_level = need_level      # How much agent needs resource
        self.owned_resources = 0
        self.moral_weight = moral_weight # How ethical the agent is (0–1)

    def selfish_utility(self, request_amount):
        # Utility based on personal gain
        return request_amount

    def fairness_utility(self, request_amount):
        # Penalize if agent takes more than fair share
        fair_share = self.model.total_resources / self.model.num_agents
        excess = max(0, (self.owned_resources + request_amount) - fair_share)
        return -excess

    def moral_utility_function(self, request_amount):
        """
        Combined moral + selfish utility
        """
        u_selfish = self.selfish_utility(request_amount)
        u_fair = self.fairness_utility(request_amount)

        # Weighted moral utility
        total_utility = (1 - self.moral_weight) * u_selfish + \
                        self.moral_weight * u_fair
        return total_utility

    def decide_request(self):
        # Agent evaluates possible request sizes
        possible_requests = [0, 1, 2, 3, 4, 5]
        utilities = {}

        for r in possible_requests:
            utilities[r] = self.moral_utility_function(r)

        # Choose request with max moral utility
        best_request = max(utilities, key=utilities.get)
        return best_request

    def step(self):
        request = self.decide_request()
        self.model.submit_request(self, request)
