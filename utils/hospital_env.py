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
        return self._get_state()

    # ----------------------------------------------------
    # Get Current State
    # ----------------------------------------------------
    def _get_state(self):
        row = self.data.iloc[self.current_index]

        return np.array([
            row["arrival_time"],                   # raw time
            row["slot_time"],                      # appointment slot
            self.current_priority if self.current_priority is not None else row["priority"],
            row["no_show_prob"],
            self.current_icu_occupied / self.icu_capacity
        ], dtype=np.float32)

    # ----------------------------------------------------
    # Environment Step
    # ----------------------------------------------------
    def step(self, action):
        row = self.data.iloc[self.current_index]

        base_priority = row["priority"]
        show = row["show"]

        # -------- Emergency Override (VISIBLE to agent) --------
        if np.random.rand() < self.emergency_probability:
            priority = 2  # critical emergency
        else:
            priority = base_priority

        self.current_priority = priority

        reward = 0

        # -------- ICU Discharge Simulation --------
        # 20% chance one patient leaves ICU per step
        if self.current_icu_occupied > 0:
            if np.random.rand() < 0.2:
                self.current_icu_occupied -= 1

        # -------- Decision Logic --------
        if action == 1:  # ACCEPT

            if show == 0:
                reward -= 5  # wasted slot
            else:
                if self.current_icu_occupied < self.icu_capacity:
                    self.current_icu_occupied += 1
                    if priority == 2:
                        reward += 20  # saving critical case
                    elif priority == 1:
                        reward += 12
                    else:
                        reward += 8
                else:
                    reward -= 20  # ICU full attempt

        else:  # REJECT

            if priority == 2:
                reward -= 25  # rejecting emergency
            elif priority == 1:
                reward -= 6
            else:
                reward -= 1

        # -------- Doctor Break Constraint --------
        if row["slot_time"] in self.doctor_break_slots and action == 1:
            reward -= 5

        # -------- Advance Time --------
        self.current_index += 1
        done = self.current_index >= self.num_patients

        next_state = None if done else self._get_state()

        return next_state, reward, done
