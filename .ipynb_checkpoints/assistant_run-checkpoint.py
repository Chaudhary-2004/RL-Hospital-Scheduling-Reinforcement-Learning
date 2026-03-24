import os
import pandas as pd
from utils.hospital_env import HospitalEnv
from utils.ai_assistant import HospitalAIAssistant

# ---------------------------------------
# Load Data
# ---------------------------------------
DATA_PATH = os.path.join("data", "synthetic_hospital_data.csv")
MODEL_PATH = os.path.join("models", "ddqn_model.pt")

data = pd.read_csv(DATA_PATH)

env = HospitalEnv(data)
assistant = HospitalAIAssistant(MODEL_PATH)

state = env.reset()
done = False

step = 0
total_reward = 0

print("\n=== AI Hospital Scheduling Started ===\n")
assistant.speak("AI Hospital Scheduling Started.")

while not done:
    step += 1

    action, explanation = assistant.decide(state)

    print(f"\nStep {step}")
    print("Decision:", explanation)

    assistant.speak(explanation)

    next_state, reward, done = env.step(action)

    print("Reward:", reward)
    print("Current ICU Load:", env.current_icu_occupied)

    total_reward += reward
    state = next_state

print("\n=== Scheduling Completed ===")
print("Total Reward:", total_reward)

assistant.speak("Scheduling completed successfully.")
assistant.speak(f"Total reward achieved is {int(total_reward)}.")
