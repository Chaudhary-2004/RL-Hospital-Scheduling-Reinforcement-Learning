import numpy as np

class HospitalEnv:
    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        self.num_patients = len(self.data)

        self.icu_capacity = 5
        self.doctor_break_slots = {120, 240}
        self.emergency_probability = 0.02

        self.reset()

    def reset(self):
        self.current_index = 0
        self.current_icu_occupied = 0
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_index]

        return np.array([
            row["arrival_time"],
            row["slot_time"],
            row["priority"],
            row["no_show_prob"],
            self.current_icu_occupied / self.icu_capacity
        ], dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.current_index]

        priority = row["priority"]
        show = row["show"]

        # Emergency override
        if np.random.rand() < self.emergency_probability:
            priority = 2

        reward = 0

        # -------- Simulate Discharge --------
        if self.current_icu_occupied > 0:
            discharge = np.random.choice([0, 1], p=[0.7, 0.3])
            self.current_icu_occupied -= discharge
            self.current_icu_occupied = max(0, self.current_icu_occupied)

        # -------- Decision --------
        if action == 1:  # Accept
            if show == 0:
                reward -= 5
            else:
                if self.current_icu_occupied < self.icu_capacity:
                    self.current_icu_occupied += 1
                    reward += 15 if priority == 2 else 10
                else:
                    reward -= 15  # ICU full penalty
        else:  # Reject
            if priority == 2:
                reward -= 20
            elif priority == 1:
                reward -= 5
            else:
                reward -= 1

        # Doctor break penalty
        if row["slot_time"] in self.doctor_break_slots and action == 1:
            reward -= 5

        self.current_index += 1
        done = self.current_index >= self.num_patients

        next_state = None if done else self._get_state()
        return next_state, reward, done
