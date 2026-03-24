import numpy as np
import pandas as pd
import random

def generate_synthetic_data(num_patients, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    DAY_MINUTES = 8 * 60

    arrival_time = np.random.randint(0, DAY_MINUTES, num_patients)
    slot_time = np.random.randint(0, DAY_MINUTES, num_patients)
    priority = np.random.choice([0, 1], size=num_patients, p=[0.8, 0.2])
    no_show_prob = np.random.uniform(0.05, 0.4, num_patients)
    show = (np.random.rand(num_patients) > no_show_prob).astype(int)

    data = pd.DataFrame({
        "patient_id": range(1, num_patients + 1),
        "arrival_time": arrival_time,
        "slot_time": slot_time,
        "priority": priority,
        "no_show_prob": no_show_prob,
        "show": show
    })

    return data
