import json
import matplotlib.pyplot as plt

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

if __name__=='__main__':
    with open("parameters.json", "r") as file:
        loaded_data = json.load(file)

    sum_obs = loaded_data['sum_obs']
    num_sensor = loaded_data['sum_sensor']
    sum_action = loaded_data['sum_action']
    action = [sublist[0] for sublist in sum_action]

    plot_num_sensor(sum_obs, num_sensor)
    #plot_num_sensor(sum_obs, action)

