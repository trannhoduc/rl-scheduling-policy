import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from filterpy.kalman import ExtendedKalmanFilter as EKF
from generate_sensors import SensorSimulator

# -----------------------STATIC PARAMETERS-------------------------------
# Define parameters
m = 20  # Number of sensors
max_m = 10
d_mean = 10  # Mean of sensor distance distribution
d_std = 2  # Standard deviation of sensor distance distribution

# Mean and standard deviation for generating scalar measurement noise covariances
cov_mean = 1e-4
cov_std = 2e-4
state_0_prob = 0.6

# Kalman filter parameters
Q = np.array([[1e-9, 0], [0, 1e-9]])
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

def hx(state, H):
    Ht = np.array(H)
    return np.dot(Ht, state)

def jacobian_fx(state):
    return np.array([
        [1 + 0.075 * np.sin(3*state[0])  , 1],
        [0.075 * np.sin(3*state[0])      , 1]
    ])

def jacobian_hx(state, H):
    Ht = np.array(H)
    return Ht

def fx():
    pass

def init(H, x, Q, R, P):
    # Initialize the EKF
    ekf = EKF(dim_x=2, dim_z=H.shape[0])

    ekf.x = x # Initial state
    ekf.P = P # State covariance
    ekf.Q = Q # Process noise
    ekf.R = R # Measurement noise

    # Assign the constant Jacobian functions and transition matrix
    ekf.F = jacobian_fx(ekf.x)
    ekf.H = jacobian_hx(ekf.x, H)

    return ekf

# Define your custom functions and classes (already provided in your snippet)
def EKF_exclude(true_state, accuracy_level):
    for i in range(len(accuracy_level)):
        noise = np.random.normal(loc=0, scale=np.sqrt(1 / 10**accuracy_level[i]))
        true_state[i] += noise
    return true_state

# Create vectorized and monitored environment with the custom wrapper
def make_custom_env():
    base_env = gym.make("MountainCarContinuous-v0")
    wrapped_env = CustomRewardWrapper(base_env)
    return Monitor(wrapped_env)

def reward_transform(original_reward, state, action):
    position, velocity = state
    transformed_reward = original_reward + 5e-6 * 0.5 * (1/10**action[1] + 1/10**action[2])
    return transformed_reward

class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.true_state = None  # Store the true state internally
        self.last_action = None
        self.ekf = init(H, x, Q, R, P)
        self.SA_simulator = SensorSimulator(m, d_mean, d_std, Q, cov_mean, cov_std, state_0_prob)
        # Extend action space to include sensor parameters
        self.action_space = Box(
            low=np.array([-1.0, np.log10(100), np.log10(500)]),  # Logarithmic scale
            high=np.array([1.0, np.log10(1e20), np.log10(1e20)]),
            dtype=np.float32,
        )

    def step(self, action):
        control_action = np.array([action[0]])  # Extract control action
        # Perform the step with the control action and retrieve the true state
        true_state, original_reward, terminated, truncated, info = self.env.step(control_action)
        self.true_state = true_state  # Update the internal true state
        self.last_action = action

        # Estimate the state based on the action's accuracy levels
        # estimated_state = EKF_exclude(np.copy(true_state), action[1:])
        estimated_state, Qs = state_estimation_function(np.copy(true_state), action[1:], self.ekf, self.SA_simulator)

        # Return the estimated state to the agent
        return estimated_state, self.reward(original_reward), terminated, truncated, info, Qs

    def reset(self, **kwargs):
        # Reset the environment and retrieve the true state
        true_state, info = self.env.reset(**kwargs)
        self.true_state = true_state  # Store the true state internally
        self.last_action = None
        return true_state, info

    def reward(self, reward):
        # Transform the reward based on the last state and action
        return reward_transform(reward, self.true_state, self.last_action)

def state_estimation_function(states, eta, ekf, SA_simulator):
    global Ps, H, Qs, R, L
    eta = 10**eta
    Qs = [] # Sensor Agents (Index of sensors)
    Ps = np.arange(m)

    ekf.F = jacobian_fx(ekf.x)  # Update F
    ekf.H = jacobian_hx(ekf.x, H)      # Update H
    ekf.predict()

    P_pr = ekf.P
    x_pr = ekf.x

    H = [] # Measurement matix

    # Check the Age of Loop of the features in state
    for i in range (len(ekf.x)):
        if L[i] > delta_L[i] and len(Qs) < max_m:
            # Filter indices where the observed state equals i
            indices = [idx for idx, state in enumerate(SA_simulator.observed_states) if state == i]
            # Intersection with Ps, create new min_index
            valid_indices = np.intersect1d(indices, Ps)

            if len(valid_indices) > 0:
                # Find the index of the sensor with the minimum distance among the valid sensors
                min_index = min(valid_indices, key=lambda idx: SA_simulator.distances[idx])
                print(f"Sensor with shortest distance observing state {i} is at index {min_index} with distance {SA_simulator.distances[min_index]}")

                if i == 0:
                    H.append([1, 0])
                else:
                    H.append([0, 1])
                ekf.H = np.array(H)  # Correct assignment to ekf.H

                # Remove that sensor from available list
                Ps = Ps[Ps != min_index]

                # Update list of sensors
                Qs.append(min_index)
                ekf.R = np.diag(SA_simulator.SA_cov_list[Qs])
            else:
                print(f"No sensors observe state {i}.")

    stop = [0,0]

    while any(ekf.P[i, i] > np.minimum(xi[i], 1/eta[i]) for i in range(len(ekf.x))) and len(Qs) < max_m and stop != [1, 1]:
        if ekf.P[0,0] <= np.minimum(xi[0], 1/eta[0]) and ekf.P[1,1] <= np.minimum(xi[1], 1/eta[1]):
            print("The requirements of covariance are fullfil!")
            break
        for i in range(len(ekf.x)):
            if ekf.P[i, i] > np.minimum(xi[i], 1/eta[i]):
                # Get indices of sensors that collect data for state i
                indices = [idx for idx, state in enumerate(SA_simulator.observed_states) if state == i]

                # Intersection with Ps, create new min_index
                valid_indices = np.intersect1d(indices, Ps)

                if len(valid_indices) > 0:
                    # Find the sensor index with the smallest covariance among the valid indices
                    min_index = min(valid_indices, key=lambda idx: SA_simulator.SA_cov_list[idx])
                    print(f"Sensor observing state {i} with smallest covariance is at index {min_index}")
                    print(f"Covariance: {SA_simulator.SA_cov_list[min_index]}")

                    # Update measurements matrix (H)
                    if i == 0:
                        H.append([1, 0])
                    else:
                        H.append([0, 1])
                    ekf.H = np.array(H)

                    # Remove that sensor from available list
                    #Ps.remove(min_index)
                    Ps = Ps[Ps != min_index]

                    # Update list of sensors
                    Qs.append(min_index)
                    ekf.R = np.diag([SA_simulator.SA_cov_list[q] for q in Qs])

                    # Update the estimation
                    sensor_values = SA_simulator.generate_sensor_values(states)
                    #ekf.update([sensor_values[i] for i in Qs], HJacobian=jacobian_hx, args=(H), Hx=hx, hx_args=(H)) # Update the EKF with the new sensor data
                    ekf.P = P_pr
                    ekf.x = x_pr
                    ekf.update(
                        np.array([sensor_values[i] for i in Qs]),  # Make sure z is a NumPy array
                        HJacobian=jacobian_hx,
                        args=(H,),  # Correct the tuple syntax here with a comma
                        Hx=hx,
                        hx_args=(H,)
                    )
                    print(f'The Update of x {ekf.x}')
                else:
                    print(f"No sensors are observing state {i}.")
                    stop[i] = 1
            else:
                stop[i] = 1

        if all(stop):
            break

    return ekf.x, Qs
