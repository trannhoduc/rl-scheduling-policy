from custom_env import *
from stable_baselines3 import PPO
import json

if __name__ == '__main__':
    # Import the pre-trained model
    model_name = "ppo_mountaincar.zip"
    model = PPO.load(model_name)
    # Create the modified environment
    env = gym.make('MountainCarContinuous-v0', render_mode='human')  # Added render_mode='rgb_array'
    env = CustomRewardWrapper(env)
    # Variable to store data
    steps = 0
    sum_sensor = []
    sum_obs    = []
    sum_action = []
    sum_P = []

    # Test the model
    for episode in range(1):  # Test for 5 episodes
        obs, _ = env.reset()  # Updated reset

        sum_obs.append(obs)
        #sum_sensor.append(len(Qs))

        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            sum_action.append(action)

            obs, reward, terminated, truncated, info, Qs, P = env.step(action)  # Updated step
            done = terminated or truncated

            total_reward += reward
            sum_obs.append(obs)
            sum_P.append(P)
            sum_sensor.append(len(Qs))
            steps += 1

            print(f'Step: {steps}, action: {action}')

        print(f"Episode {episode + 1} - Total Reward: {total_reward}")
        print(f"Episode {episode + 1} - Total Steps: {steps}")
        #input("Press Enter to proceed to the next episode...")

    # Close the environment
    env.close()

    # Convert NumPy arrays to lists for JSON serialization
    data = {
        "sum_sensor": sum_sensor,
        'sum_P': [P_n.tolist() for P_n in sum_P],
        "sum_obs": [obs.tolist() for obs in sum_obs],
        "sum_action": [action.tolist() for action in sum_action]
    }

    # Save to JSON file
    with open("parameters.json", "w") as file:
        json.dump(data, file, indent=4)

    print("Data successfully saved to parameters.json")


