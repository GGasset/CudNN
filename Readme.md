# Socket docs

connects to "\0NN_manager"

- message format:
    - action enum as size t
    * actions
        1. construct
            a) layer count : size t
            for i in layer count
                NN::ConnectionTypes : size t
                NN::NeuronTypes : size t
                neuron count : size t
                ActivationFunctions : size t
            input length : size t
            stateful : bool
            
            returns created network id
        
# Technologies used

 - LSTM architecture
	![LSTM architecture](https://i.sstatic.net/RHNrZ.jpg) 