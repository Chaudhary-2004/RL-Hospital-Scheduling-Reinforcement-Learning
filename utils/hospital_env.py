import numpy as np

class HospitalEnv:
    """
    AI-Ready ICU Scheduling Environment
    ------------------------------------
    State:
        [arrival_time, slot_time, priority, no_show_prob, icu_ratio]

    Actions:
        0 = Reject
        1 = Accept
    """

    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        self.num_patients = len(self.data)

        # System Constraints
        self.icu_capacity = 5
        self.doctor_break_slots = {120, 240}
        self.emergency_probability = 0.02

        self.reset()

    # ----------------------------------------------------
    # Reset Environment
    # ----------------------------------------------------
    def reset(self):
        self.current_index = 0
        self.current_icu_occupied = 0
        self.current_priority = None

        # Metrics tracking
        self.total_waiting_time = 0
        self.patient_count = 0
        self.busy_resource_time = 0
        self.total_resource_time = 0

        return self._get_state()

    # ----------------------------------------------------
    # Get Current State
    # ----------------------------------------------------
    def _get_state(self):
        row = self.data.iloc[self.current_index]

        return np.array([
            row["arrival_time"],
            row["slot_time"],
            self.current_priority if self.current_priority is not None else row["priority"],
            row["no_show_prob"],
            self.current_icu_occupied / self.icu_capacity
        ], dtype=np.float32)

    # ----------------------------------------------------
    # Environment Step
    # ----------------------------------------------------
    def step(self, action):
        row = self.data.iloc[self.current_index]

        waiting_time = max(0, row["slot_time"] - row["arrival_time"])
        self.total_waiting_time += waiting_time
        self.patient_count += 1

        base_priority = row["priority"]
        show = row["show"]

        # Emergency override
        if np.random.rand() < self.emergency_probability:
            priority = 2
        else:
            priority = base_priority

        self.current_priority = priority

        reward = 0

        # ICU discharge simulation
        if self.current_icu_occupied > 0:
            if np.random.rand() < 0.2:
                self.current_icu_occupied -= 1

        # Decision logic
        if action == 1:  # ACCEPT
            if show == 0:
                reward -= 5
            else:
                if self.current_icu_occupied < self.icu_capacity:
                    self.current_icu_occupied += 1
                    if priority == 2:
                        reward += 20
                    elif priority == 1:
                        reward += 12
                    else:
                        reward += 8
                else:
                    reward -= 20
        else:  # REJECT
            if priority == 2:
                reward -= 25
            elif priority == 1:
                reward -= 6
            else:
                reward -= 1

        # Doctor break constraint
        if row["slot_time"] in self.doctor_break_slots and action == 1:
            reward -= 5

        # Resource utilization tracking
        self.busy_resource_time += self.current_icu_occupied
        self.total_resource_time += self.icu_capacity

        # Advance time
        self.current_index += 1
        done = self.current_index >= self.num_patients

        next_state = None if done else self._get_state()

        # Info dictionary for training logs
        info = {
            "waiting_time": waiting_time,
            "icu_occupied": self.current_icu_occupied,
            "resource_util": (self.current_icu_occupied / self.icu_capacity) * 100
        }

        return next_state, reward, done, info

    # ----------------------------------------------------
    # Metrics after episode
    # ----------------------------------------------------
    def get_metrics(self):
        avg_wait = self.total_waiting_time / max(self.patient_count, 1)
        utilization = (self.busy_resource_time / max(self.total_resource_time, 1)) * 100
        return avg_wait, utilization