# Elastic Weight Consolidation 

This is a Python implementation of the EWC algorithm as proposed by Kirkpatrick et al, 2017      
Loosely inspired by Ari Seff's implementation for learning purposes.

## Evaluation
Below a performance comparison between networks trained either with pure SGD or additionally with EWC, evaluated on two permuted mnist tasks. 

<img src="performance_sgd.png" alt="Network without EWC" width="400"/> <img src="performance_ewc.png" alt="Network with EWC" width="400"/>
<!-- ![Network without EWC](performance_sgd.png =100x)    
![Network with EWC](performance_ewc.png =100x) -->

