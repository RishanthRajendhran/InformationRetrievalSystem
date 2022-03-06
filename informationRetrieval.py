from util import *

# Add your import statements here
from itertools import chain

class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = None

		#Fill in code here
		words = list(set(list(chain.from_iterable(list(chain.from_iterable(docs))))))

		documentIndex = {}
		invertedIndex = {}
		docsRepr = {}
		for word in words:
			invertedIndex[word] = {"docFreq":0, "containingDocs":[]}
		for i in range(len(docs)):
			documentIndex[docIDs[i]] = list(chain.from_iterable(docs[i]))
			for word in list(set(documentIndex[docIDs[i]])):
				invertedIndex[word]["docFreq"] += 1
				invertedIndex[word]["containingDocs"].append(docIDs[i])
		
		#Better to precompute doc repr because time saving >>> storage 
		for i in range(len(docs)):
			docRepr = [0]*len(words)
			# normalisationFactor = 0
			for j in range(len(words)):
				if words[j] in list(set(documentIndex[docIDs[i]])):
					termFreq = documentIndex[docIDs[i]].count(words[j])/len(documentIndex[docIDs[i]])
					docFreq = np.log(len(docs)/invertedIndex[words[j]]["docFreq"])
					docRepr[j] = termFreq*docFreq
					# normalisationFactor += ((termFreq)**2)*((docFreq)**2)
			# docsRepr[docIDs[i]] = np.array(docRepr)/normalisationFactor 
			if np.linalg.norm(docRepr) != 0:
				docsRepr[docIDs[i]] = docRepr/np.linalg.norm(docRepr)

		index = {
			"words": words,
			"documentIndex": documentIndex,
			"invertedIndex": invertedIndex,
			"docsRepr": docsRepr,
		}

		self.index = index


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here
		for query in queries:
			wordsInQuery = list(chain.from_iterable(query))
			queryRepr = [0]*len(list(self.index["invertedIndex"].keys()))
			docsRelToQuery = []
			for word in list(set(wordsInQuery)):
				if word not in list(self.index["invertedIndex"].keys()):
				# if word not in self.index["words"]:
					continue
				docsRelToQuery.extend(self.index["invertedIndex"][word]["containingDocs"])
				termFreq = (wordsInQuery.count(word))/(len(wordsInQuery))
				docFreq = np.log(len(list(self.index["documentIndex"].keys()))/self.index["invertedIndex"][word]["docFreq"])
				queryRepr[list(self.index["invertedIndex"].keys()).index(word)] = termFreq*docFreq
			queryRepr /= np.linalg.norm(queryRepr)
			docsRelToQuery = list(set(docsRelToQuery))
			docsRelToQueryRepr = []
			for (ind, doc) in enumerate(docsRelToQuery):
				# docRepr = [0]*len(self.index["invertedIndex"])
				# for (wIndex, word) in enumerate(self.index["documentIndex"][doc]):
				# 	termFreq = self.index["documentIndex"][doc].count(word)/len(self.index["documentIndex"][doc])
				# 	docFreq = np.log(len(list(self.index["documentIndex"].keys()))/self.index["invertedIndex"][word]["docFreq"])
				# 	docRepr[ind] = termFreq*docFreq
				# docRepr /= np.linalg.norm(docRepr)
				docRepr = self.index["docsRepr"][doc]
				docsRelToQueryRepr.append(docRepr)

			# print(f"{queries.index(query)}/{len(queries)}")
			cosSims = np.dot(queryRepr, np.transpose(docsRelToQueryRepr))
			doc_IDs_ordered.append([docsRelToQuery[idPos] for idPos in np.argsort(cosSims)[::-1]])
			# doc_IDs_ordered.append([list(self.index["docsRepr"].keys())[rank] for rank in np.argsort(cosSims)[::-1]])
	
		return doc_IDs_ordered




