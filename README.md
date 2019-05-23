# LEDynamics

Exploration of RNN's and research into solving the EVG problem from a dynamics perspective.

# Usage
Currently only copytask is fully functional.
```
python copytask.py [args]
```
Options:
- net-type : type of RNN to use in test
- hidden-size: number of hidden units in the network
- labels: number of labels to use for copy task
- c-length: length of copying sequence
- T: length of delay between sequence copy and output
- random-seed: seeds random number generators
- cuda: turn cuda on or off (default on)
- save: save network/activation/gradient information
- save-freq: frequency to save train/test loss accuracy and network gradient information in epochs
- LED-freq: frequency to calculate Lyupanov Exponents for saving
- p-detach: probability of detaching for DRNN and QDRNN

Available networks:
- RNN - vanilla RNN
- LSTM - vanilla LSTM
- SRNN - RNN shrink 1.0
- DRNN - random detach RNN
- QDRNN - detach based on QR decomposition




# WIPs
- Refactoring code to use torch autograd Functions for all RNN types to save and modify all aspects of gradients.
- Adding addtask
- Adding sequential MNIST
