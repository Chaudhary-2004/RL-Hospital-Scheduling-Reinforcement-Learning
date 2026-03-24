import torch
import numpy as np
import pyttsx3
from utils.models import DQN  # 5-dim trained model


def normalize_state(state):
    arrival, slot, priority, no_show, icu_ratio = state
    return np.array([
        arrival / 480.0,
        slot / 480.0,
        priority / 2.0,
        no_show,
        icu_ratio
    ], dtype=np.float32)


class HospitalAIAssistant:

    def __init__(self, model_path):
        self.model = DQN()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Text-to-Speech Engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 170)

    # --------------------------------------------------
    # Decision Function
    # --------------------------------------------------
    def decide(self, state):
        state_n = normalize_state(state)

        with torch.no_grad():
            state_tensor = torch.tensor(state_n).float().unsqueeze(0)
            q_vals = self.model(state_tensor)

            action = torch.argmax(q_vals, dim=1).item()

        explanation = self.explain_decision(state, q_vals, action)

        return action, explanation

    # --------------------------------------------------
    # Explain Decision (Model-Based Explanation)
    # --------------------------------------------------
    def explain_decision(self, state, q_vals, action):
        arrival, slot, priority, no_show, icu_ratio = state

        accept_q = q_vals[0][1].item()
        reject_q = q_vals[0][0].item()

        confidence_gap = abs(accept_q - reject_q)

        if action == 1:
            decision_text = "Patient accepted."
        else:
            decision_text = "Patient rejected."

        # Confidence level
        if confidence_gap > 2:
            confidence_text = " The decision confidence is high."
        elif confidence_gap > 0.5:
            confidence_text = " The decision confidence is moderate."
        else:
            confidence_text = " The decision confidence is low."

        # Context reasoning
        context_text = ""

        if priority >= 1:
            context_text += " High priority case detected."

        if icu_ratio > 0.8:
            context_text += " ICU capacity is critically high."

        if no_show > 0.3:
            context_text += " Patient has elevated no-show probability."

        explanation = (
            f"{decision_text}"
            f"{confidence_text}"
            f"{context_text}"
            f" (Q_accept={accept_q:.2f}, Q_reject={reject_q:.2f})"
        )

        return explanation

    # --------------------------------------------------
    # Voice Output
    # --------------------------------------------------
    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
