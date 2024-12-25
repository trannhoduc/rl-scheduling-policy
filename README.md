
# Scheduling Policy Optimization using Reinforcement Learning and Extended Kalman Filtering

Partial re-implementation of paper [Timely Communication from Sensors for Wireless Networked Control in Cloud-Based Digital Twins](https://arxiv.org/abs/2408.10241)

### Purposes

Implement a system where we have a primary agent (PA) which includes n features of the system.
Additionally, there are m sensor agents which send the data to the server to estimate the state of the system.
At the server, we use Extended Kalman Filter to estimate the status of the sytem then make a new decision to control the system (motor, robot, etc).
The paper and this project aim to use Reinforcement Learning and proposed an algorithm to select a particular number of sensor to update the data
to reduce the bandwitch consumption, energy consumption of the whole system.

1. Every query interval, the digital twin/cloud/server/edge intelligence (or anything you call it) consider the Age of Loop and Variance of each features in the system.
   Then, based on these information, it require more sensor to send the data to import Age of Loop (Information freshness) and Variance (Accuracy) of the system.
   Finally, the server send the control signal to update scheduling policy and take action at the end devices.
3. At the server, Extended Kalman Filter was used to estimate the status of the system from data garthering from SA.
   Suppose that the information of the sensors are not trustworthy and have difference accuracy.
4. To make the decision, Proximal policy optimization (PPO) was used to train the system and give the control decision.

> Mountain-Car-Continuous-v0 environment (gymnasium) was used to represent the system where the car have to choose the right velocity to reach the destination.
You can gain more information on [Gymnasisum's webpage](https://gymnasium.farama.org/)

> Temporarily, I skip Age of Loop and Resource Allocation consideration. You can see the code for AoL in this repo, however, you can found that I set the AoL of all features equal to 0 all the time.

### Algorithm
The process that the server consider Age of Loop (Information freshness) and Variance (Accuracy) to make the new scheduling policy is described as below:

<div align="center">
<img width="541" alt="image" src="https://github.com/user-attachments/assets/dc62d4bb-168e-4699-a89c-934aa4bb8a7b" />
</div>


> **NOTE**: Please read the original paper for more information.

### What is included in this project 

- 🛳️ Custom environment of Mountain Car Continuous *(custom_env.py)*
  * Add two more feature in the action space (accuracy level of postion and velocity of the car)
  * Modify the reward function which the bonus reward will higher when we have lower accuracy level and vice versa
- 🎯 Extended Kalman Filter, which use FilterPy library *(custom_env.py)*
- 🤖 State estimation function, which use algorithm above to consider and update scheduling policy, then return the estimation for the system. *(custom_env.py)*
- 🔄 Sensor's data generator, create the value of m sensor every query interval *(generate_sensors.py)*
- ✅  A pre-trained model of Proximal policy optimization (PPO) for the system. Honestly, it is not really good as the original study.
  However, it is good enough to work well and show the proper insights. *(ppo_mountaincar.zip)*
- 📊 A code to generate visualization, not too much things here, it might be updated later. *(visualize.py)*
  `Containerfile` is a more open standard for building container images than Dockerfile, you can use buildah or docker with this file.
- 🧪 A sample result which was generated before *(parameters.json)*
- 📃 Requirements file for essential packages. Indeed, I generate by put all of packages in my virtual env in it, thus, there might be some redundant packages.
  However, it not too many, so do not worry about using it.
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

> **NOTE**: The result might not be the same with the original paper due the the pre-trained model.

Thank you for your visiting !
