import numpy as np
from database import Database,CostModel,QueryHandler
from bidict import bidict
from queryVector import QueryVector
import pandas as pd
import functools

joinTypes = ['Hash','Nested','SortMerge']

# Dynammic Programming solution and training data generation
class ExhaustiveSearch():
	def __init__(self,database):
		self.relations = bidict() # Relation to Index BiDirectional Mapping
		self.joins = dict() 
		self.minCosts = [] # DP Table storing costs
		self.tuples = [] # DP Table storing tuples
		self.relationToColumn = dict() # Relation to column involved in the join
		self.database = database
		self.minDecision = []
		self.partialJoins = dict()
		self.trainingData = []
		self.relationCount = 0

	def buildPrereq(self,joins):
		for join in joins:
			column1,column2 = join
			if column1 not in self.joins.keys():
				self.joins[column1] = set()
			if column2 not in self.joins.keys():
				self.joins[column2] = set()
			self.joins[column1].add(column2)
			self.joins[column2].add(column1)
			for table in (column1[0],column2[0]):
				if table not in self.relationToColumn.keys():
					self.relationToColumn[table] = set()
					self.relations[table] = self.relationCount
					self.relationCount = self.relationCount + 1
			self.relationToColumn[column1[0]].add(column1[1])
			self.relationToColumn[column2[0]].add(column2[1])
		self.minCosts = [np.inf for i in range(np.power(2,self.relationCount))]
		self.minDecision = [np.inf for i in range(np.power(2,self.relationCount))]
		self.tuples = [0 for i in range(np.power(2,self.relationCount))]

	def _subsetGenerator(self,number):
		rep = list(np.binary_repr(number))
		ones = []
		for i,bit in enumerate(rep):
			if bit == "1":
				ones.append(i)
				rep[i] = "0"
		for i in range(1,np.power(2,len(ones))-1):
			temp = np.binary_repr(i,len(ones))
			for j in range(len(temp)):
				rep[ones[j]] = temp[j]
			yield int("".join(rep),2)

	def _fillingGenerator(self):
		numberToOnes = dict()
		onesToNumber = [[] for i in range(len(self.relations))]
		numberToOnes[1] = 1
		onesToNumber[0].append(1)
		for i in range(2,np.power(2,len(self.relations))):
			numberToOnes[i] = i%2 + numberToOnes[int(i/2)]
			onesToNumber[numberToOnes[i]-1].append(i)
		del numberToOnes
		for i in range(0,len(self.relations)):
			for num in onesToNumber[i]:
				yield(num)

	def _combineCost(self,subset1,subset2,superset):
		if self.tuples[subset1] == 0 or self.tuples[subset2] == 0:
			return 0
		costs = [CostModel.getCost(self.tuples[subset1],self.tuples[subset2],self.tuples[superset],e) for e in joinTypes]
		return min(costs)

	def _getTupleCount(self,subset): # Get number of tuples after completing a join on this subset
		rep = list(np.binary_repr(subset,self.relationCount))
		if rep.count("1") == 1:
			table = self.relations.inverse[rep.index("1")]
			return self.database.getRelation(table).tupleCount
		for i,bit in enumerate(rep):
			if bit == "1":
				rep[i] = "0"
				break
		complement = subset - int("".join(rep),2)
		tables1,tables2 = set(),set()
		for i,bit in enumerate(rep):
			if bit == "1":
				tables1.add(self.relations.inverse[i])
		for i,bit in enumerate(np.binary_repr(complement,self.relationCount)):
			if bit == "1":
				tables2.add(self.relations.inverse[i])
		finalSelectivity = 1
		for table in tables1:
			for column in self.relationToColumn[table]:
				for joinColumn in self.joins[(table,column)]:
					if joinColumn[0] in tables2:
						finalSelectivity = finalSelectivity*self.database.getSelectivity((table,column),joinColumn)
		return int(finalSelectivity*self.tuples[subset-complement]*self.tuples[complement])

	def buildPartialJoins(self):
		for val in self._fillingGenerator():
			rep = np.binary_repr(val,len(self.relations))
			self.partialJoins[val] = set()
			for i,bit in enumerate(rep):
				if bit == "1":
					table1 = self.relations.inverse[i]
					for column1 in self.relationToColumn[table1]:
						for table2,column2 in self.joins[(table1,column1)]:
							if rep[self.relations[table2]] == "0":
								continue
							if ((table2,column2),(table1,column1)) in self.partialJoins[val]:
								continue
							self.partialJoins[val].add(((table1,column1),(table2,column2)))
			self.partialJoins[val] = list(self.partialJoins[val])

	def DP(self):
		self.buildPartialJoins()
		for i in range(len(self.relations)):
			self.minCosts[np.power(2,i)] = 0
			self.minDecision[np.power(2,i)] = -1
		for val in self._fillingGenerator():
			self.tuples[val] = self._getTupleCount(val)
			if len(self.partialJoins[val]) == 0: # No joins to be done
				continue
			for subproblem in self._subsetGenerator(val):
				if len(self.partialJoins[subproblem]) == 0: # No joins
					rep = np.binary_repr(subproblem,self.relationCount)
					if int(functools.reduce(lambda a,b:int(a)+int(b),rep)) != 1:
						continue
				if len(self.partialJoins[val-subproblem]) == 0: # No joins
					rep = np.binary_repr(val-subproblem,self.relationCount)
					if int(functools.reduce(lambda a,b:int(a)+int(b),rep)) != 1:
						continue
				temp = set(self.partialJoins[val].copy())
				for join in self.partialJoins[subproblem]:
					temp.discard(join)
				for join in self.partialJoins[val-subproblem]:
					temp.discard(join)
				queryGraph = QueryVector(self.database)
				queryGraph.buildVector(self.partialJoins[val])
				queryGraph.makeJoins(self.partialJoins[subproblem])
				combineCosts = self._combineCost(subproblem,val-subproblem,val) # Combine these two subsets
				calculatedCost =  self.minCosts[subproblem]+self.minCosts[val-subproblem]+combineCosts # Cost taken for the subproblem
				if len(temp) != 0: # This was a cross join (Dont want this data)
					self.trainingData.append(list(queryGraph.featureVector))
					self.trainingData[-1].append(calculatedCost-self.minCosts[subproblem])
				if calculatedCost < self.minCosts[val]:
					self.minCosts[val] = calculatedCost
					self.minDecision[val] = subproblem

	def describeOptimalPath(self,subset):
		if int(functools.reduce(lambda a,b:int(a)+int(b),np.binary_repr(subset))) == 1:
			return
		self.describeOptimalPath(self.minDecision[subset])
		self.describeOptimalPath(subset-self.minDecision[subset])
		leftTable = ""
		for i,bit in enumerate(np.binary_repr(self.minDecision[subset],self.relationCount)):
			if bit == "1":
				leftTable = leftTable + self.relations.inverse[i]
		rightTable = ""
		for i,bit in enumerate(np.binary_repr(subset-self.minDecision[subset],self.relationCount)):
			if bit == "1":
				rightTable = rightTable + self.relations.inverse[i]
		print("Merging "+leftTable + " with "+rightTable + " cost " + str(self.minCosts[subset]-self.minCosts[self.minDecision[subset]]-self.minCosts[subset-self.minDecision[subset]]))	

	def dumpTrainingData(self,filename):
		dataframe = pd.DataFrame(np.array(self.trainingData))
		with open(filename,'w') as f:
			dataframe.to_csv(f,header=False,index=None)

# Left-Deep solution

class LeftDeep(ExhaustiveSearch):
	def __init__(self,database):
		super(LeftDeep,self).__init__(database)
		self.minLDCosts = []
		self.minLDDecision = []

	def buildPrereq(self,joins):
		super(LeftDeep,self).buildPrereq(joins)
		self.minLDCosts = [np.inf  for i in range(np.power(2,len(self.relations)))]
		self.minLDDecision = [np.inf for i in range(np.power(2,len(self.relations)))]

	def _subsetGenerator(self,val):
		rep = list(np.binary_repr(val))
		for i,bit in enumerate(rep):
			if bit == "1":
				rep[i] = "0"
				yield int("".join(rep),2)
				rep[i] = "1"

	def LD(self):
		self.buildPartialJoins()
		for i in range(len(self.relations)):
			self.minLDCosts[np.power(2,i)] = 0
			self.minLDDecision[np.power(2,i)] = -1
		for val in self._fillingGenerator():
			self.tuples[val] = self._getTupleCount(val)
			if len(self.partialJoins[val]) == 0: # No joins to be done
				continue
			for subproblem in self._subsetGenerator(val):
				if len(self.partialJoins[subproblem]) == 0: # No joins
					rep = np.binary_repr(subproblem,self.relationCount)
					if functools.reduce(lambda a,b:int(a)+int(b),rep) != 1:
						continue
				if len(self.partialJoins[val-subproblem]) == 0: # No joins
					rep = np.binary_repr(val-subproblem,self.relationCount)
					if functools.reduce(lambda a,b:int(a)+int(b),rep) != 1:
						continue
				combineCosts = self._combineCost(subproblem,val-subproblem,val)
				calculatedCost =  self.minLDCosts[subproblem]+self.minLDCosts[val-subproblem]+combineCosts # Cost taken for the subproblem
				if calculatedCost < self.minLDCosts[val]:
					self.minLDCosts[val] = calculatedCost
					self.minLDDecision[val] = subproblem


database = Database()
database.loadDatabase("Database1/database.txt")
exhaustiveSearch = ExhaustiveSearch(database)
query = QueryHandler.loadQuery("Database1/query1")
exhaustiveSearch.buildPrereq(query)
exhaustiveSearch.DP()
exhaustiveSearch.dumpTrainingData("ExtractedData/query1")
exhaustiveSearch.describeOptimalPath(np.power(2,exhaustiveSearch.relationCount)-1)
print("Optimal Cost:",exhaustiveSearch.minCosts[-1])