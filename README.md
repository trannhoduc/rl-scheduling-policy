
# Scheduling Policy Optimization using Reinforcement Learning and Extended Kalman Filtering

Partial re-implementation of paper [Timely Communication from Sensors for Wireless Networked Control in Cloud-Based Digital Twins](https://arxiv.org/abs/2408.10241)

### Purposes

Implement a system where we have a primary agent (PA) which includes n features of the system.
Additionally, there are m sensor agents which send the data to the server to estimate the state of the system.
At the server, we use the Extended Kalman Filter to estimate the status of the system and then make a new decision to control the system (motor, robot, etc).
The paper and this project aim to use Reinforcement Learning and propose an algorithm to select a particular number of sensors to update the data
to reduce the bandwidth consumption and energy consumption of the whole system.

1. Every query interval, the digital twin/cloud/server/edge intelligence (or anything you call it) considers the Age of Loop and the Variance of each feature in the system.
   Then, based on this information, it requires more sensors to send data to import the system's Age of Loop (Information freshness) and Variance (Accuracy).
   Finally, the server sends the control signal to update the  scheduling policy and take action at the end devices.
3. At the server, an Extended Kalman Filter was used to estimate the status of the system from data gathered from SA.
   Suppose that the sensors' information is not trustworthy and has different accuracy.
4. To make the decision, Proximal policy optimization (PPO) was used to train the system and give the control decision.

> The Mountain-Car-Continuous-v0 environment (gymnasium) was used to represent the system in which the car must choose the right velocity to reach its destination.
You can gain more information on [Gymnasisum's webpage](https://gymnasium.farama.org/)

> Temporarily, I skipped the Resource Allocation consideration.

### Algorithm
The process that the server considers Age of Loop (Information freshness) and Variance (Accuracy) to make the new scheduling policy is described below:

<div align="center">
<img width="541" alt="image" src="https://github.com/user-attachments/assets/dc62d4bb-168e-4699-a89c-934aa4bb8a7b" />
</div>


> **NOTE**: Please read the original paper for more information.

### What is included in this project 

- 🛳️ Custom environment of Mountain Car Continuous *(custom_env.py)*
  * Add two more features in the action space (accuracy level of position and velocity of the car)
  * Modify the reward function so that the bonus reward will be higher when we have a lower accuracy level and vice versa
- 🎯 Extended Kalman Filter, which uses FilterPy library *(custom_env.py)*
- 🤖 State estimation function, which uses the algorithm above to consider and update scheduling policy, then returns the estimation for the system. *(custom_env.py)*
- 🔄 Sensor's data generator, create the value of m sensor every query interval *(generate_sensors.py)*
- ✅  A pre-trained model of Proximal policy optimization (PPO) for the system. Honestly, it is not really good as the original study.
  However, it is good enough to work well and show the proper insights. *(ppo_mountaincar.zip)*
- 📊 A code to generate visualization, not too many things here, it might be updated later. *(visualize.py)*
- 🧪 A sample result which was generated before *(parameters.json)*
- 📃 Requirements file for essential packages. Indeed, I generated it by putting all of the packages in my virtual env in it, thus, there might be some redundant packages.
  However, it is not too many, so do not worry about using it. *(requirements.txt)*
- 🤖 Finally, the main function where we simulate the model and make the car run its journey. *(main.py)*

> My contact: nhoduc.tran@miun.se

---
# How to use it

## Install packages

```bash
pip install -r requirements.txt
```

## Usage

```py
python main.py
```

## Visulization

```py
python visualize.py
```

> **NOTE**: The result might not be the same as the original paper due the the pre-trained model.

Thank you for your visit!
