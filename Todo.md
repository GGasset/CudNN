# Todo

- ~~LSTM~~
- ~~Neuron~~
- ~~NEAT connections~~ 
	- ~~Execute~~
	- ~~Gradients~~
    - ~~Optimize~~
        * ~~Various methods~~
        * ~~Evolution methods~~
- ~~adaptative learning rates~~
    - ~~return cost while training~~
- ~~reinforcement learning cost function~~
    - ~~GAE in GPU~~
        CPU function that given a value function estimator NN and rewards, computes GAE 
    - Proximal Policy optimization cost function
- ~~server socket~~
    - ~~Create log file~~
    - Create destructor for socket interpreter -> NN manager
    - Use poll instead of select
    - Improve security
        1. Bind to a file for security
        2. Use script for server init
            * Create file
            * Set chmod for socket file
            * Set max open fd for server
            * Start server
            * with flag --strict-security or -s
                - reboot

- ~~client_socket~~
- socket functions
    - ~~construct -- destruct~~
    - Get ID of pointer to a NN
        you pass a id and returns another id that is a reference to a existing NN
        it has its own execution data to train in parallel
    - training execute
    - training functions
    - save & load
    - evolution methods
    - Inference
    - delete memory

- Create logging options for training in csv format
    - Add python script for plotting it

- Paralelize layer derivative calculation calls


# Bugs
 - ~~device array being used as host array in NeuronLayer and LSTMLayer remove neuron~~
 - ~~bug in add neuron~~
 - constrain evolution
    * if removing neuron in a layer with 1 neuron remove layer (now it won't delete the neuron)
 - ~~constrain weights and biases to prevent nans (reset NaNs)~~

### Not to do until all Layer constructors are made:

- ~~Tensorflow-like class constructor~~
- ~~Save, Load, Cloning~~
- Crossover
- ~~Evolution~~
- ~~Modify kernel launches to have more capacity of neurons (current max 1024)~~
