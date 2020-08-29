# Gym-Query
## A minimal OpenAI gym environment for query optimizers in databases

This repository models a database, relations and queries to be used for reinforcement learning tasks, similar to [Learning to Optimize Join Queries With Deep
Reinforcement Learning](https://arxiv.org/pdf/1808.03196.pdf). This work was done as a part of the course COL868 (Machine Learning for Databases) at IIT Delhi, under Prof Maya Ramanath.

### Schematics

#### gym-query
The gym-query folder contains an OpenAI gym implementation of an environment characterised by a join query, relations and a database. The initial state consists of a database with various tables corresponding to a particular query with differing tuple counts. The environment models the action space as the set of all possible joins of two tables. The cost of each action i.e a particular join, is calculated using the cost-model specified in the database and the selectivity of the result of the join. Any query optimizer can work in this environment, and can learn how to specify which joins to perform first. A random query optimizer is provided in queryOptimizer.py

#### database.py

This file models an entire database with relations, cost model for various joins and creating and loading database files. A sample database is provided in the folder Database1 along with various join queries. 

#### benchmarks.py 




