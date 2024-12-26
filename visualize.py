import json
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

def plot_combined_chart(mean_position, reverb_uncertainty, DT_requirement, RL_requirement, num_SAs):
    """
    Plots a combined chart with:
    - Mean position and REVERB uncertainty (shaded region).
    - DT and RL requirements (upper and lower bounds calculated using mean_position ± requirement).
    - Number of selected SAs as a bar chart.

    Args:
        mean_position (array): Mean position values.
        reverb_uncertainty (array): Uncertainty values for the shaded region around mean position.
        DT_requirement (array): DT uncertainty values.
        RL_requirement (array): RL uncertainty values.
        num_SAs (array): Number of selected SAs (for the bar chart).
    """
    # Generate QI (Operational Index) automatically
    QI = np.arange(len(mean_position))

    # Convert inputs to NumPy arrays for element-wise operations
    mean_position = np.array(mean_position)
    reverb_uncertainty = np.array(reverb_uncertainty)
    DT_requirement = np.array(DT_requirement)
    RL_requirement = np.array(RL_requirement)
    num_SAs = np.array(num_SAs)

    # Calculate upper and lower bounds for DT and RL requirements
    DT_upper = mean_position + DT_requirement
    DT_lower = mean_position - DT_requirement
    RL_upper = mean_position + RL_requirement
    RL_lower = mean_position - RL_requirement

    # Plot the data
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot mean position with REVERB uncertainty
    ax1.plot(QI, mean_position, 'b--', label="Mean position", linewidth=1.5)
    ax1.fill_between(QI, mean_position - reverb_uncertainty, mean_position + reverb_uncertainty,
                     color='gray', alpha=0.4, label="REVERB uncertainty")

    # Plot DT requirement (upper and lower bounds)
    ax1.fill_between(QI, DT_lower, DT_upper, color='green', alpha=0.4, label="DT requirement")

    # Plot RL requirement (upper and lower bounds)
    ax1.fill_between(QI, RL_lower, RL_upper, color='red', alpha=0.3, label="RL requirement")

    # Configure Primary Y-axis
    ax1.set_xlabel("QI (Index)")
    ax1.set_ylabel("Position", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc="upper left", fontsize=10)

    # Add Secondary Y-axis for number of SAs
    ax2 = ax1.twinx()
    bar_width = 0.8  # Set bar width for separation
    ax2.bar(QI, num_SAs, width=bar_width, color='orange', alpha=0.7, label="Number of selected SAs")
    ax2.set_ylabel("No. selected SAs", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(0, 20)  # Set y-axis range to [0, 20]

    # Add Secondary Y-axis legend
    ax2.legend(loc="upper right", fontsize=10)

    # Final adjustments
    plt.tight_layout()
    plt.show()

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

    position_level = [sublist[1] for sublist in sum_action]
    position_level = np.array(position_level)  # If it's not already a NumPy array
    position_level_values = np.sqrt(1 / (10 ** position_level))

    position = [sublist[0] for sublist in sum_obs]
    P_0 = [sublist[0][0] for sublist in sum_P]
    P_1 = [sublist[1][1] for sublist in sum_P]

    print(position_level_values)

    #plot_num_sensor(sum_obs, num_sensor)
    #plot_num_sensor(sum_obs, action)
    #plot_num_sensor(sum_obs, P_0)
    #plot_heatmap(1, 'Action')
    plot_combined_chart(position, P_0, [np.sqrt(0.01)] * len(num_sensor), position_level_values, num_sensor)

