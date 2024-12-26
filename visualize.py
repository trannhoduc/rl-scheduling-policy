import json
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

def plot_num_sensor(sum_obs, num_sensor):
    # Extract positions and velocities from sum_obs
    positions = [obs[0] for obs in sum_obs]
    velocities = [obs[1] for obs in sum_obs]

    # Ensure num_sensor, positions, and velocities have the same length
    if len(num_sensor) != len(positions):
        min_length = min(len(num_sensor), len(positions))
        num_sensor = num_sensor[:min_length]
        positions = positions[:min_length]
        velocities = velocities[:min_length]

    # Create a scatter plot with color representing the number of sensors
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(positions, velocities, c=num_sensor, cmap='viridis', s=100, edgecolor='k')
    plt.title('Position vs Velocity with Number of Sensors as Color')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.colorbar(scatter, label='Number of Sensors')
    plt.grid(True)
    plt.show()

def plot_heatmap(index, title):
    '''
    Plot the heat map between the position, velocity
    and action / uncertainty level of position/velocity.
    - Index = 0: action
    - Index = 1: accuracy level of position
    - Index = 2: accuracy level of velocity
    '''
    model_name = "ppo_mountaincar.zip"
    model = PPO.load(model_name)
    # Create a grid of positions and velocities
    grid_x, grid_y = np.meshgrid(
        np.linspace(-1.2, 0.6, 100),  # x-axis: position range
        np.linspace(-0.07, 0.07, 100)  # y-axis: velocity range
    )
    # Compute uncertainties for each grid point
    uncertainty_grid = np.array([
        [model.predict([pos, vel], deterministic=True)[0][index] for pos, vel in zip(row_x, row_y)]
        for row_x, row_y in zip(grid_x, grid_y)
    ])

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(uncertainty_grid, extent=[-1.2, 0.6, -0.07, 0.07], origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(heatmap, label="Level")
    plt.title(f"{title} Heatmap")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    with open("parameters.json", "r") as file:
        loaded_data = json.load(file)

    sum_obs = loaded_data['sum_obs']
    sum_P = loaded_data['sum_P']
    num_sensor = loaded_data['sum_sensor']
    sum_action = loaded_data['sum_action']

    action = [sublist[0] for sublist in sum_action]
    P_0 = [sublist[0][0] for sublist in sum_P]
    P_1 = [sublist[1][1] for sublist in sum_P]

    #plot_num_sensor(sum_obs, num_sensor)
    #plot_num_sensor(sum_obs, action)
    #plot_num_sensor(sum_obs, P_0)
    plot_heatmap(1, 'Action')

