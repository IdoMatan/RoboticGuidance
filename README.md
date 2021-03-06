# RL Autonomous Guidance of Human Controlled Vehicle
> An Actor-Critic algorithm for training an autonomous drone towards guiding a human controlled car

> *Based on Microsoft AirSim simulator and UnrealEngine*

> Final Project in Technion Course 096235 - Intelligent Interactive Systems
---
![drone_and_car](https://github.com/IdoMatan/RoboticGuidance/blob/master/images/airsim_1.png)
---
## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Playing a Game](#playing-a-game)
- [Display results](#display-results)
- [Results](#results)
- [License](#license)

---
## Dependencies
- AirSim simulator set up to support both drone and car together (see installation for more details)


---
## Installation
> unfortunately setting Microsoft AirSim up is a bit of a hassle as the Airsim current release doesnt support
both drone and car simultaneously so needs to be slighly modified to allow it.
- Follow [this medium guide](https://medium.com/@idoglanz/setting-up-microsoft-airsim-to-simulate-a-drone-and-car-together-708079b2d0f?sk=a31cfc18e2fe1948874bc0dadd80c182) to setup the AirSim with both a drone and a car
- Clone [this repo](https://github.com/IdoMatan/RoboticGuidance.git) to the same parent folder of the Airsim plugin
- If using Conda, create an environment using `conda env create -f airsim_env.yaml`

---
## Playing a Game
- Open the blocks.uproject in `./AirSim/Unreal/Environments/Block` and press 'PLAY'
- Run `run.py`, return to simulator and press the screen to allow manual control of car
- Once the drone starts moving, follow it with the car using the keyboard arrow keys

---
## Display results
- Open the `RoboticGuidance.ipynb` jupyter notebook
- Run the different cells to plot both individual and accumulated results

---
## Results
- A sample episode of the training phase. One of the first things the drone learned was to slow down if the car is far away

![reward_graph](https://github.com/IdoMatan/RoboticGuidance/blob/master/images/RewardGraph.jpeg)
- A short gif of a training run (5x speed)

![episode_gif](https://github.com/IdoMatan/RoboticGuidance/blob/master/images/GuidanceSample.gif)

- Trajectory plot (drone in red, car in blue)

![trajectory_gif](https://github.com/IdoMatan/RoboticGuidance/blob/master/images/example.gif)


---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
