# ReinforcementLearningProject
This it the Git Repository for the exam project in the course 02465 Introduction to Reinforcement Learning and Control by the group
- Peter Grønning, s183922
- Johannes Boe Reiche, s175549
- Mads Christian Berggrein Andersen, s173934
- Sunniva Olsrud Punsvik, s183924

In this project the classic control methods iterative Lineare Quadratic Regulator (iLQR) and Differential Dynamic Programming (DDP) are tested in a stochastic Cartpole enviroment with and withouth the feedback control stategy Model Predictive Control (MPC).


## Overview of Repository
The folder "iLQR" contains the implementation of the iLQR algorithm. These files are reused from excersises in week 3, with slight modifications in LQR making it more 1-to-1, with the algorithm proposed in Tassa12.
LQR is further extended with an argument for DDP instead of iLQR.

The enviroments used are dp_cartpole_env and dp_symbolic_env from week 3 as well.
Runge Kutta 4 integrations has been added to the cartpole_env and Hessians for the second derivatives of systems dynamics for DDP has been added to the symbolic env. 


Cartpolebalancing.py (from week 5 edited to our experiment) is the file in which the experiments have been conducted. get_values.py, load_trajectories.py and results.py are all for evaluating the experiments.
test_experiments.py is a short script for rendering a trajectory from the folder "Trajectories".

The folder 'utils' include helper scripts from the course excercises.









