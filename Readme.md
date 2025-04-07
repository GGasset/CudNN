# Overview
## What?

* CudNN is an AI framework made from scratch using CUDA/C++
* Does not support transformers, and there is no plan for it.
* See [why](#why) to find out the niche of this project.

## Technologies used
### Planned-WIP
 - L1, L2 regularization (Planned)
 - Adam optimizer (Planned)
 - C++ Socket bindings (WIP)
 - [PPO](https://arxiv.org/abs/1707.06347) Reinforcement Learning Policy Gradient Method used in ChatGPT (Planned)
 - [MinLSTM](https://arxiv.org/abs/2410.01201) Simpler versions of LSTM (Planned)

### Working
 - [GAE](https://arxiv.org/abs/1506.02438) Generalized Advantage Estimator
 - LSTM architecture
	![LSTM architecture](https://i.sstatic.net/RHNrZ.jpg) 

## How?
* It has been my 7th iteration on an AI framework, and I wouldn't say I've learnt a lot otherwise, I've learnt a lot.
	* I've learnt how to create scalable solutions thanks to trial-error (Sadly and Proudly).
	* Conditioning is optional, Conditions are controlled, Consistency is key.

## Why?
* CudNN trades inference efficiency for model structure flexibility.
	* That means that continous training and also using evolution methods is faster on CudNN than it is on Tensorflow, you would do that to find an optimal model size, i.e. best structure for hidden layers.
		* As per my knowledge, to add neurons on Tensorflow in realtime during training, to, lets say, find the most efficient model size for a given task, you would need to:
    		1. Save the model on disk
    		2. Load it as text
    		4. Modify it
    		5. Save it
    		6. Load into a model (Class).
* CudNN has a WIP client API which doesn't require CUDA compilation, being useful for use with common game engines wich make hard compiling Cuda.
* In the future, it will feature godot bindings for its socket client, C# bindings for Unity are not planned, UE already uses c++.

#### Godspeed
