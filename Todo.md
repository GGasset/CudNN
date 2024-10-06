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
    - Proximal Policy optimization cost function
- ~~server socket~~
    - Create destructor for socket interpreter -> NN manager
    - Use poll instead of select
    - Improve security
        1. Bind to a file for security
        2. Use script for server init
            * Create file
            * Set chmod for server
            * Set max open fd for server
            * Start server
            * with flag --strict-security or -s
                - sudo reboot
                    (only use sudo for this)
- ~~client_socket~~
- socket functions
    - ~~construct -- destruct~~
    - training execute
    - training functions
    - save & load
    - evolution methods
    - Inference

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
