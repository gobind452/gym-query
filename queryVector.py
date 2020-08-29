from database import Database,CostModel,QueryHandler
import numpy as np

class QueryVector():
	def __init__(self,database):
		self.database = database
		self.featureVector = np.zeros(self.database.columnCount*(self.database.columnCount+1)).astype(int)
		self.joins = []
		self.currentCost = 0
		self.columnCount = self.database.columnCount

	def _getIndex(self,type,index):
		if type == 'ToDo': # Joins to do
			index1,index2 = min(index),max(index)
			indexInVector = self.columnCount*(self.columnCount+1)/2 - (self.columnCount-index1)*(self.columnCount-index1+1)/2 + index2-(index1)
			return int(indexInVector)
		if type == 'Done': # Joins done
			index1,index2 = min(index),max(index)
			indexInVector = self.columnCount*(self.columnCount+1) - (self.columnCount-index1)*(self.columnCount-index1+1)/2 + index2-(index1)
			return int(indexInVector)
	
	def buildVector(self,joins): # Build a query vector
		self.joins = joins[:]
		for join in joins:
			column1,column2 = join
			index1,index2 = self.database.columnIndex.inverse[column1],self.database.columnIndex.inverse[column2]
			indexInVector = self._getIndex('ToDo',(index1,index2))
			self.featureVector[indexInVector] = 1

	def makeJoins(self,joins): # Merges relations, and changes the feature vector accordingly
		for join in joins:
			column1,column2 = join
			index1,index2 = self.database.columnIndex.inverse[column1],self.database.columnIndex.inverse[column2]
			indexInVector = self._getIndex('ToDo',(index1,index2))
			self.featureVector[indexInVector] = 0
			indexInVector = self._getIndex('Done',(index1,index2))
			self.featureVector[indexInVector] = 1