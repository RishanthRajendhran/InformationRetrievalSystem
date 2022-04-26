from util import *

# Add your import statements here




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		precision = np.sum([True if query_doc_IDs_ordered[i] in true_doc_IDs else False for i in range(min(len(query_doc_IDs_ordered),k))])/k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		precision = 0
		for q in range(len(query_ids)):
			relDocs = []
			positions = [1,2,3,4]
			pos = 0
			while len(relDocs) <= k and pos < len(positions):
				relDocs.extend([int(doc["id"]) for doc in qrels if doc["query_num"] == str(query_ids[q]) and doc["position"] == positions[pos]])
				pos += 1
			precision += self.queryPrecision(doc_IDs_ordered[q], query_ids[q], relDocs, k)
		meanPrecision = precision/len(query_ids)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		recall = np.sum([True if query_doc_IDs_ordered[i] in true_doc_IDs else False for i in range(min(len(query_doc_IDs_ordered), k))])/len(true_doc_IDs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		recall = 0
		for q in range(len(query_ids)):
			relDocs = []
			positions = [1,2,3,4]
			pos = 0
			while len(relDocs) <= k and pos < len(positions):
				relDocs.extend([int(doc["id"]) for doc in qrels if doc["query_num"] == str(query_ids[q]) and doc["position"] == positions[pos]])
				pos += 1
			recall += self.queryRecall(doc_IDs_ordered[q], query_ids[q], relDocs, k)
		meanRecall = recall/len(query_ids)

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if precision == 0 or recall == 0:
			fscore = 0
		else:
			fscore = (2*precision*recall)/(precision+recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		fscore = 0
		for q in range(len(query_ids)):
			relDocs = []
			positions = [1,2,3,4]
			pos = 0
			while len(relDocs) <= k and pos < len(positions):
				relDocs.extend([int(doc["id"]) for doc in qrels if doc["query_num"] == str(query_ids[q]) and doc["position"] == positions[pos]])
				pos += 1
			fscore += self.queryFscore(doc_IDs_ordered[q], query_ids[q], relDocs, k)
		meanFscore = fscore/len(query_ids)

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		DCG = 0
		for i in range(min(min(len(query_doc_IDs_ordered), len(true_doc_IDs)), k)):
			relI = 0
			for j in range(len(true_doc_IDs)):
				if true_doc_IDs[j][0] == query_doc_IDs_ordered[i]:
					relI = true_doc_IDs[j][1]
					break
			DCG += relI/np.log2((i+1)+1)
		
		IDCG = 0
		for i in range(min(len(true_doc_IDs), k)):
			IDCG += true_doc_IDs[i][1]/np.log2((i+1)+1)

		if IDCG != 0:
			nDCG = DCG/IDCG
		else:
			nDCG = 0

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		NDCG = 0
		for q in range(len(query_ids)):
			relDocs = []
			positions = [1,2,3,4]
			pos = 0
			while len(relDocs) <= k and pos < len(positions):
				relDocs.extend([(int(doc["id"]), positions[pos]) for doc in qrels if doc["query_num"] == str(query_ids[q]) and doc["position"] == positions[pos]])
				pos += 1
			NDCG += self.queryNDCG(doc_IDs_ordered[q], query_ids[q], relDocs, k)
		meanNDCG = NDCG/len(query_ids)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		avgPrecision = 0
		for i in range(1, k+1):
			avgPrecision += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i)
		avgPrecision /= k

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		averagePrecision = 0
		for q in range(len(query_ids)):
			relDocs = []
			positions = [1,2,3,4]
			pos = 0
			while len(relDocs) <= k and pos < len(positions):
				relDocs.extend([int(doc["id"]) for doc in q_rels if doc["query_num"] == str(query_ids[q]) and doc["position"] == positions[pos]])
				pos += 1
			averagePrecision += self.queryAveragePrecision(doc_IDs_ordered[q], query_ids[q], relDocs, k)
		meanAveragePrecision = averagePrecision/len(query_ids)

		return meanAveragePrecision

