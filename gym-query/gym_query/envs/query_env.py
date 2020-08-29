import gym 
from gym import error,spaces,utils
from gym.utils import seeding
from queryVector import QueryVector
from bidict import bidict
from database import CostModel
from copy import deepcopy
import gc

joinTypes = ['Hash','Nested','SortMerge']

# Defines a OpenAI Query Environment for testing out RL algorithms

class QueryEnv(gym.Env):
	def __init__(self):
		pass

	def init(self,database,query):
		self.queryVector = QueryVector(database)
		self.queryVector.buildVector(query)
		self.actions = dict()
		self.database = database
		self.vertexCount = 0
		self.vertexInfo = dict()
		self.tuples = dict()
		self.vertexIndexes = list()
		for join in query:
			column1,column2 = join
			table1,att1 = column1
			table2,att2 = column2
			if table1 not in self.vertexInfo:
				self.vertexInfo[table1] = set()
				self.vertexCount=self.vertexCount+1
				self.tuples[table1] = self.database.getRelation(table1).tupleCount
			if table2 not in self.vertexInfo:
				self.vertexCount = self.vertexCount+1
				self.vertexInfo[table2] = set()
				self.tuples[table2] = self.database.getRelation(table2).tupleCount
			if (table1,table2) not in self.actions and (table2,table1) not in self.actions:
				self.actions[(table1,table2)] = set()
				self.actions[(table1,table2)].add(join)
			elif (table1,table2) not in self.actions:
				self.actions[(table2,table1)].add(join)
			self.vertexInfo[table1].add(table2)
			self.vertexInfo[table2].add(table1)
		self.actionSpace = spaces.Tuple((spaces.Discrete(self.vertexCount),spaces.Discrete(self.vertexCount-1),spaces.Discrete(len(joinTypes))))
		self.vertexIndexes = list(self.vertexInfo.keys())

	def done(self):
		return self.vertexCount == 1

	def currentState(self,action):
		return self.queryVector.featureVector

	def nextState(self,action):
		index1,index2,joinType = action
		if index2 >= index1:
			index2 = index1+1
		table1,table2 = self.vertexIndexes[index1],self.vertexIndexes[index2]
		nextState = deepcopy(self.queryVector)
		if (table1,table2) in self.actions.keys():
			try:
				nextState.makeJoins(list(self.actions[(table1,table2)]))
			except KeyError:
				pass
		elif (table2,table1) in self.actions.keys():
			try:
				nextState.makeJoins(list(self.actions[(table1,table2)]))
				table1,table2 = table2,table1
			except KeyError:
				pass
		return nextState.featureVector

	def step(self,action):
		index1,index2,joinType = action
		if index2 >= index1:
			index2 = index1+1
		table1,table2 = self.vertexIndexes[index1],self.vertexIndexes[index2]
		if (table1,table2) in self.actions.keys():
			self.queryVector.makeJoins(list(self.actions[(table1,table2)]))
		elif (table2,table1) in self.actions.keys():
			self.queryVector.makeJoins(list(self.actions[(table2,table1)]))
			table1,table2 = table2,table1
		mergeTable = table1+table2
		self.vertexInfo[mergeTable] = set()
		mergeTableInfo = set()
		for i,vertex in enumerate(self.vertexIndexes):
			if vertex != table1 and vertex != table2:
				if table1 in self.vertexInfo[vertex]:
					self.vertexInfo[vertex].remove(table1)
					self.vertexInfo[vertex].add(mergeTable)
					self.actions[(mergeTable,vertex)] = set()
					if (table1,vertex) in self.actions.keys():
						self.actions[(mergeTable,vertex)].update(self.actions[(table1,vertex)])
					if (vertex,table1) in self.actions.keys():
						self.actions[(mergeTable,vertex)].update(self.actions[(vertex,table1)])
					self.vertexInfo[mergeTable].add(vertex)
				if table2 in self.vertexInfo[vertex]:
					self.vertexInfo[vertex].remove(table2)
					self.vertexInfo[vertex].add(mergeTable)
					if (mergeTable,vertex) not in self.actions.keys():
						self.actions[(mergeTable,vertex)] = set()
					if (table2,vertex) in self.actions.keys():
						self.actions[(mergeTable,vertex)].update(self.actions[(table2,vertex)])
					if (vertex,table2) in self.actions.keys():
						self.actions[(mergeTable,vertex)].update(self.actions[(vertex,table2)])
					self.vertexInfo[mergeTable].add(vertex)
		selectivity = 1
		try:
			for join in self.actions[(table1,table2)]:
				selectivity = selectivity*self.database.getSelectivity(join[0],join[1])
		except KeyError:
			pass
		self.tuples[mergeTable] = int(selectivity*self.tuples[table1]*self.tuples[table2])
		del self.tuples[table1],self.tuples[table2]
		del self.vertexInfo[table1],self.vertexInfo[table2]
		self.vertexIndexes = list(self.vertexInfo.keys())
		self.vertexCount = self.vertexCount-1
		self.actionSpace = spaces.Tuple((spaces.Discrete(self.vertexCount),spaces.Discrete(self.vertexCount-1),spaces.Discrete(len(joinTypes))))
		gc.collect()

	def cost(self,action):
		index1,index2,joinType = action
		if index2 >= index1:
			index2 = index1+1
		table1,table2 = self.vertexIndexes[index1],self.vertexIndexes[index2]
		selectivity = 1
		if (table1,table2) in self.actions.keys():
			for join in self.actions[(table1,table2)]:
				selectivity = selectivity*self.database.getSelectivity(join[0],join[1])
		elif (table2,table1) in self.actions.keys():
			for join in self.actions[(table2,table1)]:
				selectivity = selectivity*self.database.getSelectivity(join[0],join[1])
		return round(CostModel.getCost(self.tuples[table1],self.tuples[table2],self.tuples[table1]*self.tuples[table2]*selectivity,joinTypes[joinType]),3)

	def reset(self):
		self.queryVector = QueryVector(database)
		self.queryVector.buildVector(query)

	def actionSpaceIterator(self):
		space1,space2,space3 = self.actionSpace
		for i in range(space1.n):
			for j in range(space2.n):
				for k in range(space3.n):
					yield (i,j,k)

	def render(self,mode='human'):
		print("Joins left : ",self.vertexCount-1)

	def describe(self,action):
		index1,index2,joinType = action
		if index2 >= index1:
			index2 = index1+1
		table1,table2 = self.vertexIndexes[index1],self.vertexIndexes[index2]
		print("Merging "+table1+" with "+table2 + " using " + joinTypes[joinType])

	def close(self):
		pass