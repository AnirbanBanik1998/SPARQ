# Title
Mobility-aware Joint Power Control and Incremental Spectrum Re-Allocation using 
Q-Learning for D2D communication.

## Steps

* Clone the Repository https://github.com/AnirbanBanik1998/D2D.git
* For the 1st case in which D2D rx is not moving radially wrt D2D tx:
  python3 compare.py 0
* For the 2nd case in which D2D rx is moving radially wrt D2D tx:
  python3 compare.py 1
  
* To compare between the update parameters, run python3 converge.py 0 or 1
  

## Documentation

* **core** -> This directory contains the core modules for running the simulation
1. **config.py** -> This configures the cell model, the D2Ds, the Cellular UEs, 
    and Channels.
2. **utils.py** -> This contains some of the utility functions needed 
    in the algorithm.
3. **q_learn.py** -> This contains the various functions for Q-learning, 
    with different terminating conditions.
4. **display.py** -> Contains the functionalities for plotting the results.


* **algo.py** -> Contains the main algorithm defined. Power Control using Q-Learning 
    is written as a separate functionality for multiprocessing. 
* **compare.py** -> Compares the performance of mAQRP, mQRP, Open-Loop with 
    Swapping, and Open-Loop algorithms.
* **converge.py** -> Compares the convergence of mAQRP algorithm with different
    update parameters.

