# model.py
from mesa import Model
from mesa.time import RandomActivation
from agents import ResourceAgent
import numpy as np

class ResourceModel(Model):
    def __init__(self, num_agents=10, total_resources=20, moral_weight=0.5):
        self.num_agents = num_agents
        self.total_resources = total_resources
        self.available_resources = total_resources
        self.schedule = RandomActivation(self)
        self.requests = {}

        for i in range(num_agents):
            need = np.random.randint(1, 5)
            agent = ResourceAgent(i, self, need, moral_weight)
            self.schedule.add(agent)

    def submit_request(self, agent, amount):
        self.requests[agent] = amount

    def allocate_resources(self):
        """
        Greedy allocation (environment is neutral, not enforcing fairness)
        First-come-first-serve based on scheduler order.
        This allows selfish agents to hoard resources.
        """
        for agent, amt in self.requests.items():
            if self.available_resources <= 0:
                break

            allocated = min(amt, self.available_resources)
            agent.owned_resources += allocated
            self.available_resources -= allocated

    def step(self):
        self.requests = {}
        self.schedule.step()
        self.allocate_resources()
