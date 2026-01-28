# main.py
from model import ResourceModel
from metrics import fairness_index

STEPS = 10

model = ResourceModel(num_agents=10, total_resources=20, moral_weight=0.2)

for step in range(STEPS):
    model.step()

resources = [a.owned_resources for a in model.schedule.agents]
fairness = fairness_index(resources)

print("Final Resource Distribution:", resources)
print("Fairness Index:", fairness)
