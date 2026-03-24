import torch
import numpy as np
from gtts import gTTS
import tempfile
import os

from utils.models import DQN


# --------------------------------------------
# State Normalization
# --------------------------------------------
def normalize_state(state):
    arrival, slot, priority, no_show, icu_ratio = state

    return np.array([
        arrival / 480.0,
        slot / 480.0,
        priority / 2.0,
        no_show,
        icu_ratio
    ], dtype=np.float32)


# --------------------------------------------
# AI Assistant Class
# --------------------------------------------
class HospitalAIAssistant:

    def __init__(self, model_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()


    # --------------------------------------------
    # Decision Function
    # --------------------------------------------
    def decide(self, state):

        state_n = normalize_state(state)

        with torch.no_grad():
            state_tensor = torch.tensor(state_n).float().unsqueeze(0).to(self.device)
            q_vals = self.model(state_tensor)

            action = torch.argmax(q_vals, dim=1).item()

        explanation = self.explain_decision(state, q_vals, action)

        return action, explanation, q_vals.cpu().numpy()[0]


    # --------------------------------------------
    # Explain Decision
    # --------------------------------------------
    def explain_decision(self, state, q_vals, action):

        arrival, slot, priority, no_show, icu_ratio = state

        accept_q = q_vals[0][1].item()
        reject_q = q_vals[0][0].item()

        confidence_gap = abs(accept_q - reject_q)

        decision_text = "Patient accepted." if action == 1 else "Patient rejected."

        # Confidence
        if confidence_gap > 2:
            confidence_text = " Decision confidence is high."
        elif confidence_gap > 0.5:
            confidence_text = " Decision confidence is moderate."
        else:
            confidence_text = " Decision confidence is low."

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
            f" Q_accept={accept_q:.2f}, Q_reject={reject_q:.2f}."
        )

        return explanation


    # --------------------------------------------
    # Streamlit-Safe Voice Output
    # --------------------------------------------
    def speak(self, text):
        """
        Generates MP3 file and returns path
        Streamlit will play it using st.audio()
        """

        tts = gTTS(text=text, lang='en')

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_path = temp_file.name
        temp_file.close()

        tts.save(temp_path)

        return temp_path
