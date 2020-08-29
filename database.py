from bidict import bidict
import numpy as np
import gc

# Random Choices for parameters (Similar to PostGres)

bufferSize = 1000
cpuIOCost = 0.02
cpuTupleCost = 0.01
hashingCost = 0.001
blockSize = 100
diskLoadCost = 0.1
bufferToBlock = bufferSize/blockSize
nonLinearityFactor = 1

class Relation():
	def __init__(self,name,columns,tupleCount):
		self.name = name # Name of the table
		self.columns = bidict({i:columns[i] for i in range(len(columns))}) # Bi-Directional Map from columns to index 
		self.tupleCount = tupleCount # Number of rows in the table
		self.columnCount = len(columns) # Number of columns in the table

# CostModel for two joins

class CostModel():
	def getCost(tupleCountA,tupleCountB,finalTupleCount,joinType):
		blockCountA = int(tupleCountA/blockSize)+1
		blockCountB = int(tupleCountB/blockSize)+1
		cost = 0
		if joinType == 'Nested': # Nested Loop Block join
			if blockCountA+blockCountB < bufferToBlock: # All blocks can fit in buffer
				cost = (blockCountA+blockCountB)*blockSize*diskLoadCost
				cost = cost + tupleCountA*tupleCountB*cpuTupleCost + finalTupleCount*cpuIOCost
				return cost
			else:
				cost = (blockCountA*blockCountB*diskLoadCost + blockCountA*diskLoadCost)*blockSize
				cost = cost + tupleCountA*tupleCountB*cpuTupleCost + finalTupleCount*cpuIOCost
				return cost
		if joinType == "Hash":
			if blockCountA + blockCountB < bufferToBlock:
				cost = (blockCountA+blockCountB)*blockSize*diskLoadCost
				cost = blockCountA*blockSize*hashingCost + tupleCountB*hashingCost + finalTupleCount*cpuIOCost
				return cost - nonLinearityFactor*blockCountA
			else:
				cost = (blockCountA*blockCountB*diskLoadCost + blockCountA*diskLoadCost)*blockSize + tupleCountA*hashingCost
				cost = cost + tupleCountB*hashingCost + finalTupleCount*cpuIOCost
				return cost - nonLinearityFactor*blockCountA
		if joinType == "SortMerge":
			if blockCountA+blockCountB < bufferToBlock: # All blocks can fit in buffer
				cost = (blockCountA+blockCountB)*blockSize*diskLoadCost
				cost = cost + tupleCountA*np.log(tupleCountA+1)*cpuTupleCost + tupleCountB*np.log(tupleCountB+1)*cpuTupleCost + (tupleCountA+tupleCountB)*cpuTupleCost + finalTupleCount*cpuIOCost
				return cost
			else:
				cost = blockCountA*np.log(blockCountA/bufferToBlock+1)*diskLoadCost*blockSize + blockCountB*np.log(blockCountB/bufferToBlock+1)*diskLoadCost*blockSize
				cost = cost + tupleCountA*np.log(bufferToBlock*tupleCountA/blockCountA+1)*cpuTupleCost + tupleCountB*np.log(bufferToBlock*tupleCountB/blockCountB+1)*cpuTupleCost
				cost = cost + tupleCountA*np.log(blockCountA/bufferToBlock+1)*cpuTupleCost +  tupleCountB*np.log(blockCountB/bufferToBlock+1)*cpuTupleCost
				cost = cost + finalTupleCount*cpuIOCost
				return cost

# Database simulator (Will be replaced by an interface to PostGres)

class Database():
	def __init__(self):
		self.relationList = [] # Relation Instances in the Database
		self.columnIndex = bidict({})
		self.selectivity = [] # Selectivity Matrix (Column to Column)
		self.columnCount = 0
		self.relationCount = 0
		self.relationToIndex = dict()

	def getRelation(self,name):
		return self.relationList[self.relationToIndex[name]]

	def getSelectivity(self,column1,column2):
		return self.selectivity[self.columnIndex.inverse[column1]][self.columnIndex.inverse[column2]]

	def loadDatabase(self,fileName):
		with open(fileName,'r') as f:
			lines = f.readlines()
		relationCount = int(lines[0].rstrip("\n"))
		for i in range(relationCount):
			info = lines[i+1].split(";")
			name = info[0]
			tupleCount = int(info[2])
			columns = info[1].split(',')
			for j,column in enumerate(columns):
				self.columnIndex[j+self.columnCount] = (name,column)
			self.columnCount = self.columnCount+len(columns)
			self.relationList.append(Relation(name,columns,tupleCount))
			self.relationToIndex[name] = i
			self.relationCount = self.relationCount+1
		if len(lines) == relationCount+1:
			self.selectivity = np.abs(np.random.normal(loc=0.005,scale=0.002,size=(self.columnCount,self.columnCount)))
			self.selectivity = np.add(self.selectivity,np.transpose(self.selectivity))
			self.selectivity = np.add(np.identity(self.columnCount),np.subtract(self.selectivity,np.diag(self.selectivity.diagonal())))
			return
		for i in range(self.columnCount):
			temp = lines[relationCount+1+i].rstrip("\n").split(',')
			temp = [float(e) for e in temp]
			self.selectivity.append(temp)

	def dumpDatabase(self,fileName):
		with open(fileName,'w') as f:
			f.write(str(len(self.relationList))+"\n")
			for i in range(len(self.relationList)):
				toWrite = str(self.relationList[i].name)+";"
				for j in range(len(self.relationList[i].columns)):
					toWrite = toWrite+self.relationList[i].columns[j]+","
				toWrite = toWrite[:-1]
				toWrite = toWrite+";"+str(self.relationList[i].tupleCount)
				f.write(toWrite+"\n")
			for i in range(self.columnCount):
				toWrite = ""
				for j in range(self.columnCount):
					toWrite = toWrite + str(round(self.selectivity[i][j],5))+","
				toWrite = toWrite[:-1]
				f.write(toWrite+"\n")

	def reset(self):
		self.relationList = [] # Relation Instances in the Database
		self.columnIndex = bidict({})
		self.selectivity = [] # Selectivity Matrix (Column to Column)
		self.columnCount = 0
		self.relationToIndex = dict()
		gc.collect()

	def randomiseDatabase(self,relationCount,maxColumn):
		self.reset()
		avgTupleCount = 0
		for i in range(relationCount):
			name = chr(i+65)
			tupleCount = np.random.randint(low=100,high=1001)
			avgTupleCount = tupleCount + avgTupleCount
			columnCount = np.random.randint(low=1,high=maxColumn)
			columns = []
			for j in range(columnCount):
				self.columnIndex[j+self.columnCount] = (name,chr(j+97))
				columns.append(chr(j+97))
			self.columnCount = self.columnCount+len(columns)
			self.relationList.append(Relation(name,columns,tupleCount))
			self.relationToIndex[name] = i
			self.relationCount = self.relationCount + 1
		avgTupleCount = avgTupleCount/relationCount
		self.selectivity = np.abs(np.random.normal(loc=0.1/avgTupleCount,scale=0.2/avgTupleCount,size=(self.columnCount,self.columnCount)))
		self.selectivity = np.add(self.selectivity,np.transpose(self.selectivity))
		self.selectivity = np.add(np.identity(self.columnCount),np.subtract(self.selectivity,np.diag(self.selectivity.diagonal())))

# QueryHandler simulator (Loads and processes query files into tuples)

class QueryHandler():
	def loadQuery(fileName):
		with open(fileName,'r') as f:
			lines = f.readlines()
		joins = []
		for line in lines:
			line = line.rstrip("\n")
			line = line.split(";")
			line[0] = tuple(line[0].split(","))
			line[1] = tuple(line[1].split(","))
			joins.append(tuple(line))
		return joins

	def dumpQuery(fileName,joins):
		with open(fileName,'w') as f:
			for join in joins:
				f.write(join[0][0]+","+join[0][1]+";"+join[1][0]+","+join[1][1]+"\n")

	def createRandomQueries(fileName,database,queryCount,maxJoins):
		relationCount = len(database.relationList)
		for i in range(queryCount):
			joins = []
			joinCount = np.random.randint(low=1,high=maxJoins+1)
			for _ in range(joinCount):
				choice = np.random.choice(relationCount,size=(2),replace=False)
				table1,table2 = database.relationList[choice[0]],database.relationList[choice[1]]
				column1,column2 = table1.columns[np.random.randint(low=0,high=len(table1.columns))],table2.columns[np.random.randint(low=0,high=len(table2.columns))]
				join = ((table1.name,column1),(table2.name,column2))
				if join not in joins:
					joins.append(join)
			QueryHandler.dumpQuery(fileName+str(i),joins)

if __name__ == "__main__":
	database = Database()
	database.randomiseDatabase(relationCount=5,maxColumn=5)
	database.dumpDatabase("Database1/database.txt")
	QueryHandler.createRandomQueries("Database1/query",database,100,5)