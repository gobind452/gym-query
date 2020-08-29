import gym,gym_query
from database import Database,QueryHandler
import torch
import pandas as pd
import numpy as np

database = Database()
database.loadDatabase("Database1/database.txt")
query = QueryHandler.loadQuery("Database1/query1")

queryEnv = gym.make("query-v0")
queryEnv.init(database,query)

totalCost = 0
while queryEnv.done() == False: # Greedy algorithm for picking join
	action = queryEnv.actionSpace.sample() # Samples a random join
	queryEnv.describe(action)
	totalCost = totalCost + cost # Total cost of the plan
	queryEnv.step(action) # Performs the action
	queryEnv.render()

print("Total cost:",totalCost)