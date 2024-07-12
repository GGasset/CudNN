# Todo

- ~~LSTM~~
- ~~Neuron~~
- ~~NEAT connections~~ 
	- ~~Execute~~
	- ~~Gradients~~
- ~~adaptative learning rates~~
    - ~~return cost while training~~
- ~~reinforcement learning cost functions~~

# Bugs
 - ~~device array being used as host array in NeuronLayer and LSTMLayer remove neuron~~
 - ~~bug in add neuron~~
 - constrain evolution
    * if removing neuron in a layer with 1 neuron remove layer (now it won't delete the neuron)
    * constrain weights and biases to prevent nans

### Not to do until all Layer constructors are made:

- ~~Tensorflow-like class constructor~~
- Save, Load, ~~Cloning~~
- Crossover
- ~~Evolution~~
- ~~Modify kernel launches to have more capacity of neurons (current max 1024)~~
