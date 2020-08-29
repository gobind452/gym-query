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

# LookUp Table
lookupTable = dict() # Build lookup table
tableSource = pd.read_csv("ExtractedData/query1",header=None)
for row in tableSource.values:
	cost = row[-1]
	key = list()
	for elem in row[:-1]:
		key.append(int(elem))
	key = tuple(key)
	lookupTable[key] = cost

del tableSource

def getBestJoin():
	bestCost,bestAction,bestQ = np.inf,-1,np.inf
	for action in queryEnv.actionSpaceIterator():
		cost = queryEnv.cost(action)
		nextState = tuple(queryEnv.nextState(action))
		try:
			value = lookupTable[nextState]
		except KeyError:
			queryEnv.describe(action)
			continue
		if cost+value < bestQ: # Optimal Q value action
			bestAction = action
			bestCost = cost
			bestQ = cost + value
	return bestAction,bestCost

totalCost = 0
while queryEnv.done() == False: # Greedy algorithm for picking join
	#action = queryEnv.actionSpace.sample() # Samples a random join
	action,cost = getBestJoin()
	queryEnv.describe(action)
	print(cost)
	totalCost = totalCost + cost # Total cost of the plan
	queryEnv.step(action) # Performs the action
	queryEnv.render()

print("Total cost:",totalCost)