import streamlit as st
import pandas as pd
import numpy as np
from utils.hospital_env import HospitalEnv
from ai_assistant import HospitalAIAssistant
from utils.llm_explainer import explain_decision

st.set_page_config(page_title="RL Hospital AI", layout="wide")

st.title("🏥 RL Hospital AI Scheduling Assistant")

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/synthetic_hospital_data.csv")

data = load_data()

# ============================================================
# SESSION STATE
# ============================================================

if "env" not in st.session_state:
    st.session_state.env = HospitalEnv(data)
    st.session_state.state = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.total_reward = 0

@st.cache_resource
def load_assistant():
    return HospitalAIAssistant("notebooks/models/ddqn_model.pt")

assistant = load_assistant()

# ============================================================
# 🧩 CUSTOM PATIENT INPUT PANEL
# ============================================================

st.sidebar.header("🧩 Custom Patient Input (Optional)")

use_custom = st.sidebar.checkbox("Use Custom Patient Input")

if use_custom:

    arrival = st.sidebar.slider("Arrival Time (minutes)", 0, 480, 120)
    slot = st.sidebar.slider("Slot Time (minutes)", 0, 480, 150)
    priority = st.sidebar.selectbox("Priority Level", [0, 1])
    no_show = st.sidebar.slider("No-show Probability", 0.0, 1.0, 0.2)
    icu_ratio = st.sidebar.slider("ICU Utilization Ratio", 0.0, 1.0, 0.3)

    custom_state = np.array(
        [arrival, slot, priority, no_show, icu_ratio],
        dtype=np.float32
    )

    st.session_state.state = custom_state
    st.session_state.done = False

state = st.session_state.state

# ============================================================
# TOP SECTION (2 COLUMNS)
# ============================================================

col1, col2 = st.columns(2)

if state is not None:

    arrival, slot, priority, no_show, icu_ratio = state

    with col1:
        st.subheader("📋 Patient Information")

        state_df = pd.DataFrame({
            "Feature": [
                "Arrival Time (minutes)",
                "Slot Time (minutes)",
                "Priority Level",
                "No-show Probability",
                "ICU Utilization Ratio"
            ],
            "Value": [
                arrival,
                slot,
                priority,
                round(no_show, 3),
                round(icu_ratio, 3)
            ]
        })

        st.dataframe(state_df, use_container_width=True)

        st.subheader("🏥 ICU Utilization")
        st.progress(float(icu_ratio))

        if priority == 1:
            st.info("HIGH priority patient")
        else:
            st.info("LOW priority patient")

    with col2:
        st.subheader("📊 Feature Visualization")

        plot_values = [
            arrival / 480,
            slot / 480,
            priority,
            no_show,
            icu_ratio
        ]

        plot_df = pd.DataFrame({
            "Feature": ["Arrival", "Slot", "Priority", "No-show", "ICU"],
            "Value": plot_values
        })

        st.bar_chart(plot_df.set_index("Feature"))

# ============================================================
# DECISION + REWARD SECTION
# ============================================================

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    if st.button("🤖 Make AI Decision"):

        if not st.session_state.done:

            action, explanation, q_values = assistant.decide(st.session_state.state)

            st.success(explanation)

            q_df = pd.DataFrame({
                "Action": ["Reject", "Accept"],
                "Q-Value": q_values
            })
            st.bar_chart(q_df.set_index("Action"))

            # ================= FIXED STEP FUNCTION =================
            result = st.session_state.env.step(action)

            # Handle BOTH old and new gym formats
            if len(result) == 3:
                next_state, reward, done = result
            elif len(result) == 4:
                next_state, reward, done, info = result
            elif len(result) == 5:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                raise ValueError("Unexpected number of return values from env.step()")
            # ======================================================

            llm_explanation = explain_decision(st.session_state.state, action, reward)

            st.subheader("🧠 AI Decision Explanation")
            st.write(llm_explanation)

            st.session_state.total_reward += reward
            st.session_state.state = next_state
            st.session_state.done = done

            st.write(f"Reward Received: {reward}")

        else:
            st.warning("Simulation completed. Please reset.")

with col4:
    st.subheader("💰 Reward Tracking")
    st.metric("Total Reward", st.session_state.total_reward)

    if st.button("🔄 Reset Simulation"):
        st.session_state.env = HospitalEnv(data)
        st.session_state.state = st.session_state.env.reset()
        st.session_state.done = False
        st.session_state.total_reward = 0
        st.success("Simulation Reset.")

# ============================================================
# TRAINING ANALYTICS SECTION (TABS)
# ============================================================

st.markdown("---")
st.header("📈 Training Analytics")

tab1, tab2 = st.tabs(["Episode Rewards", "Training Loss"])

with tab1:
    try:
        rewards_df = pd.read_csv("notebooks/results/dqn_rewards.csv")
        st.line_chart(rewards_df["reward"])

        moving_avg = rewards_df["reward"].rolling(10).mean()
        st.line_chart(moving_avg)

    except:
        st.info("Reward CSV not found.")

with tab2:
    try:
        losses_df = pd.read_csv("notebooks/results/dqn_losses.csv")
        st.line_chart(losses_df["loss"])
    except:
        st.info("Loss CSV not found.")

