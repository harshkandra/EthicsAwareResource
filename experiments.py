# experiments.py
from model import ResourceModel
from metrics import fairness_index
import pandas as pd

moral_weights = [0.0, 0.3, 0.5, 0.7, 1.0]
results = []

for mw in moral_weights:
    model = ResourceModel(num_agents=10, total_resources=20, moral_weight=mw)

    for _ in range(10):
        model.step()

    resources = [a.owned_resources for a in model.schedule.agents]
    fairness = fairness_index(resources)

    results.append({
        "moral_weight": mw,
        "resources": resources,
        "fairness_index": fairness
    })

df = pd.DataFrame(results)
print(df)

df.to_csv("experiment_results.csv", index=False)
