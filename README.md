# D2D
Mobility-aware Dynamic Joint Power Control and Resource Allocation for D2D 
underlaying cellular networks

## Steps

* Clone the Repository https://github.com/AnirbanBanik1998/D2D.git
* For the 1st case in which D2D rx is not moving radially wrt D2D tx:
  python3 algo.py 0
* For the 2nd case in which D2D rx is moving radially wrt D2D tx:
  python3 algo.py 1
  
  
* To compare between the caching parameters, run python3 main.py 0 or 1
  

## Documentation

* **config.py** -> This configures the cell model, the D2Ds, the Cellular UEs, 
    and Channels.
* **utils.py** -> This contains some of the utility functions needed 
    in the algorithm.
* **q_learn.py** -> This contains the various functions for Q-learning, 
    with different terminating conditions.
* **display.py** -> Contains the functionalities for plotting the results.
* **algo.py** -> Contains the main algorithm defined. Arguments reqd -> 0 or 1

