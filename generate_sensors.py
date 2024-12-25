import numpy as np

class SensorSimulator:
    def __init__(self, m, d_mean, d_std, Q, cov_mean, cov_std, state_0_prob):
        self.m = m
        self.d_mean = d_mean
        self.d_std = d_std
        self.Q = Q
        self.cov_mean = cov_mean
        self.cov_std = cov_std

        # Generate Gaussian-distributed distances
        self.distances = np.random.normal(loc=d_mean, scale=d_std, size=m)
        self.distances = np.clip(self.distances, 0, None)  # Ensure distances are non-negative

        # Generate Gaussian-distributed scalar measurement noise covariances
        self.SA_cov_list = self._generate_scalar_covariances()

        # Randomly assign sensors to observe state 0 or state 1
        self.observed_states = self._assign_observed_states(state_0_prob)

        # Generate randomly the AoL
        self.aol = self._generate_aol()

    def _generate_aol(self,):
        return 2 * self.distances / (self.d_mean / 2) + 0.1

    def _generate_scalar_covariances(self):
        covariances = np.random.normal(loc=self.cov_mean, scale=self.cov_std, size=self.m)
        return np.abs(covariances)

    def _assign_observed_states(self, state_0_prob):
        return np.random.choice([0, 1], size=self.m, p=[state_0_prob, 1 - state_0_prob])

    def generate_sensor_values(self, true_state):
        if true_state.shape != (2,):
            raise ValueError("The true state must have shape (2,).")

        sensor_values = []
        for i in range(self.m):
            # Generate scalar Gaussian noise for measurement and process noise
            measurement_noise = np.random.normal(loc=0, scale=np.sqrt(self.SA_cov_list[i]))
            process_noise = np.random.normal(loc=0, scale=np.sqrt(self.Q[self.observed_states[i], self.observed_states[i]]))
            # Compute sensor value for the observed state
            observed_value = true_state[self.observed_states[i]] + process_noise + measurement_noise
            sensor_values.append(observed_value)

        return sensor_values

    def get_sensor_distances(self):
        return self.distances

    def get_observed_states(self):
        return self.observed_states

if __name__ == '__main__':
    # -----------------------STATIC PARAMETERS-------------------------------
    # Define parameters
    m = 20  # Number of sensors
    max_m = 1
    d_mean = 10  # Mean of sensor distance distribution
    d_std = 2  # Standard deviation of sensor distance distribution

    # Mean and standard deviation for generating scalar measurement noise covariances
    cov_mean = 1
    cov_std = 2
    state_0_prob = 0.6

    # Kalman filter parameters
    Q = np.array([[1e-2, 0], [0, 1e-2]])
    x = np.array([0, 0])  # Initial state
    P = np.array([[1, 0], [0, 1]]) * 100000  # State covariance
    R = np.array([[2e-8, 0], [0, 2e-8]])
    Qs = []  # Sensor Agents (Index of sensors)
    H = np.array([[1, 0], [0, 1]])
    Ps = np.arange(m)

    # Age of loop threshold
    L = np.array([0, 0])

    # ---------------------CAN BE CHANGED-------------------
    delta_L = np.array([5, 5])
    xi = np.array([0.01, 0.002])

    SA_simulator = SensorSimulator(m, d_mean, d_std, Q, cov_mean, cov_std, state_0_prob)

    print(SA_simulator.get_sensor_distances())
    print(SA_simulator.SA_cov_list)