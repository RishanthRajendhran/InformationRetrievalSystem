from util import *

# Add your import statements here
from itertools import chain

class InformationRetrieval():

	def __init__(self, k):
		self.index = None
		self.k = k

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
		bigrams = []
		for doc in docs:
			bigramInDoc = list(nltk.bigrams(list(chain.from_iterable(doc))))
			bigrams.extend(bigramInDoc)
		bigrams = list(set(bigrams))

		documentIndex = {}
		invertedIndex = {}
		docsRepr = {}

		#Initialise inverted index for bigrams: number of documents the bigram occurs in 
		#and list of docIDs the bigram occurs in
		for bigr in bigrams:
			invertedIndex[bigr] = {"docFreq":0, "containingDocs":[]}

		#Populate the inverted index for words intialised above
		#Go over each document and incement the docCounts for each word found in that document
		#Also attack the corresponding docIDs to the list associated with those words
		for i in range(len(docs)):
			#Flatten the document which is stored as a list of list of words in each sentence 
			#and store the bigrams 
			documentIndex[docIDs[i]] = list(nltk.bigrams(list(chain.from_iterable(docs[i]))))
			for bigr in list(set(documentIndex[docIDs[i]])):
				invertedIndex[bigr]["docFreq"] += 1
				invertedIndex[bigr]["containingDocs"].append(docIDs[i])
		

		# DFs = []
		# for bigr in list(invertedIndex.keys()):
		# 	DFs.append(invertedIndex[bigr]["docFreq"])
		# # DFs = np.sort(list(set(DFs)))
		# # print(DFs[::-1])
		# print(np.mean(DFs))
		# exit(0)

		#Remove bigrams which appear in too less documents
		toRem = []
		for bigr in list(invertedIndex.keys()):
			if invertedIndex[bigr]["docFreq"] < 2:
				if random.uniform(0,1) <= 0.95:
					toRem.append(bigr) 

		for tR in toRem:
			del invertedIndex[tR]
			bigrams.remove(tR)

		print(len(list(invertedIndex.keys())))
		# exit(0)
		
		#Better to precompute document representation because time saving >>> storage 
		for i in range(len(docs)):
			docRepr = [0]*len(bigrams)
			for bigr in list(set(documentIndex[docIDs[i]])):
				if bigr not in bigrams:
					continue
				termFreq = documentIndex[docIDs[i]].count(bigr)/len(documentIndex[docIDs[i]])
				docFreq = np.log(len(docs)/invertedIndex[bigr]["docFreq"])
				docRepr[bigrams.index(bigr)] = termFreq*docFreq 
			docsRepr[docIDs[i]] = docRepr


		docsTerms = list(docsRepr.values())
		termsDocs = np.transpose(docsTerms).tolist()
		#T - term matrix in terms of latent concepts
		#S - singular values associated with each latent concept
		#Dh - transpose of the document matrix in terms of latent concepts
		T, S, Dh = np.linalg.svd(termsDocs)
		
		#k-rank approximation
		t = T[:, 0:self.k].copy()
		s = S[0:self.k].copy()
		dh = Dh[0:self.k,:].copy()


		#Normalise document representations to nullify effect of lengths of documents
		for j in range(len(dh[0])):
			if np.linalg.norm(dh[:,j]) != 0:
				dh[:,j] /= np.linalg.norm(dh[:,j])
		

		index = {
			"bigrams": bigrams,
			"documentIndex": documentIndex,
			"invertedIndex": invertedIndex,
			"docsRepr": docsRepr,
			"termsDocs": termsDocs,
			"t": t,
			"s": s,
			"dh": dh,
			"docIDs": docIDs,
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
			bigramsInQuery = list(nltk.bigrams(list(chain.from_iterable(query))))
			queryRepr = [0]*len(list(self.index["invertedIndex"].keys()))
			for bigram in list(set(bigramsInQuery)):
				if bigram not in list(self.index["invertedIndex"].keys()):
					continue
				termFreq = (bigramsInQuery.count(bigram))/(len(bigramsInQuery))
				docFreq = np.log(len(list(self.index["documentIndex"].keys()))/self.index["invertedIndex"][bigram]["docFreq"])
				queryRepr[list(self.index["invertedIndex"].keys()).index(bigram)] = termFreq*docFreq

			pseudoDoc = np.dot(queryRepr, np.dot(self.index["t"], np.linalg.inv(np.diag(self.index["s"]))))
			if np.linalg.norm(pseudoDoc) != 0:
				pseudoDoc /= np.linalg.norm(pseudoDoc)

			cosSims = []
			cosSimsIndex = []
			for d in range(len(self.index["dh"].T)):
				cosSim = np.dot(self.index["dh"].T[d],pseudoDoc)
				cosSims.append(cosSim)
				cosSimsIndex.append(self.index["docIDs"][d])
			doc_IDs_ordered.append([id for _, id in sorted(zip(cosSims, cosSimsIndex),reverse=True)])	

			# cosSims = np.dot(self.index["dh"].T,pseudoDoc)
			# cosSimsSort = np.argsort(cosSims)

			# cur_doc_IDs_ordered = [0]*len(self.index["docIDs"])
			# for i in range(len(cosSimsSort)):
			# 	cur_doc_IDs_ordered[cosSimsSort[i]] = self.index["docIDs"][i]
			# doc_IDs_ordered.append(cur_doc_IDs_ordered)

			# print(cosSims)
			# cosSims = np.argpartition(cosSims, -len(cosSims))[-len(cosSims):]
			# print(cosSims)
			# exit(0)
			# # doc_IDs_ordered.append(cosSims)
			# # doc_IDs_ordered.append(np.array(self.index["docIDs"])[cosSims])
			# # print("1)")
			# # print(doc_IDs_ordered)
			# # doc_IDs_ordered = []

			# docsRelToQuery = list(set(docsRelToQuery))
			# docsRelToQueryRepr = []
			# for (ind, doc) in enumerate(docsRelToQuery):
			# 	# docRepr = [0]*len(self.index["invertedIndex"])
			# 	# for (wIndex, word) in enumerate(self.index["documentIndex"][doc]):
			# 	# 	termFreq = self.index["documentIndex"][doc].count(word)/len(self.index["documentIndex"][doc])
			# 	# 	docFreq = np.log(len(list(self.index["documentIndex"].keys()))/self.index["invertedIndex"][word]["docFreq"])
			# 	# 	docRepr[ind] = termFreq*docFreq
			# 	# docRepr /= np.linalg.norm(docRepr)
			# 	docRepr = self.index["docsRepr"][doc]
			# 	docsRelToQueryRepr.append(docRepr)

			# cosSims = np.dot(queryRepr, np.transpose(docsRelToQueryRepr))
			# doc_IDs_ordered.append([docsRelToQuery[idPos] for idPos in np.argsort(cosSims)[::-1]])
			## doc_IDs_ordered.append([list(self.index["docsRepr"].keys())[rank] for rank in np.argsort(cosSims)[::-1]])
		return doc_IDs_ordered




