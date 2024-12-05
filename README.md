# CS6910 Assignment 3

> :warning: This implementation doesn't work as intended and has some major issues that must be fixed. Please refrain from using it as is.

## Author: Mathew Thomas - CS22M056

Link to `wandb` report: https://wandb.ai/cs22m056/cs6910-assignment-3/reports/CS6910-Assignment-3--Vmlldzo0NDI0Nzc2

### Training and Evaluation of the model.

Please run the file `run.py` to start running and evaluating the model. This file takes one command line argument, which tells the script whether to use the _attention_ model or _vannila_ model. Run `python3 run.py` to run the vanilla model with the best hyperparameters. To run the attention model, run `python3 run.py attn`.

### Files

1. `main.py` - Contains the vanilla model implementation.
2. `atten.py` - Contains the attention model implementation.
3. `sweep.py` - Contains sweep-related parameters.
4. `run.py` - Used for running and evaluating the model from the command line.
5. `aux.py` - Auxillary functions
6. `data.py` - Contains the implementation for data preprocessing.
