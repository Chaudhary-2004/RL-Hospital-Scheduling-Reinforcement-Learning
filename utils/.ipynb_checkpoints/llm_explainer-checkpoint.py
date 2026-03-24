import os
import pandas as pd
from datetime import datetime

RESULT_PATH = "results/assistant/llm_decision_explanations.csv"


def explain_decision(state, action, reward):

    arrival, slot, priority, no_show, icu_ratio = state

    decision = "ACCEPT" if action == 1 else "REJECT"

    explanation = f"""
Decision: {decision}

Patient arrival time: {arrival} minutes
Scheduled slot time: {slot} minutes

Priority level: {priority}
No-show probability: {round(no_show,3)}
ICU utilization ratio: {round(icu_ratio,3)}

The RL agent evaluated expected reward and selected
the action with the highest Q-value while balancing
patient priority and ICU capacity.

Observed reward from environment: {reward}
"""

    log_explanation(state, action, reward, explanation)

    return explanation


def log_explanation(state, action, reward, explanation):

    log = {
        "timestamp": datetime.now(),
        "state": str(state),
        "action": action,
        "reward": reward,
        "explanation": explanation
    }

    df = pd.DataFrame([log])

    if os.path.exists(RESULT_PATH):
        df.to_csv(RESULT_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(RESULT_PATH, index=False)