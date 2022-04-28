from util import *

# Add your import statements here
from itertools import chain

k1HP = 0.3
k2HP = 0.9
b1HP = 0.025

# number of clusters
K_clusters= 200
sim_digree=0.3
######### Clustering

def findCosSim(a, b):
	return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def spherical_kmeans(n_clusters_, list_of_words_vectors,sim_digree):
	embeddings_matrix_csr = csr_matrix(list_of_words_vectors)
	spherical_kmeans = SphericalKMeans( max_similar=sim_digree, init='similar_cut', 
		n_clusters = n_clusters_)
	labels = spherical_kmeans.fit_predict(embeddings_matrix_csr)
	centers = spherical_kmeans.cluster_centers_
	return labels , centers

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
		newDocs = []
		lemmatizer = WordNetLemmatizer()

		for doc in docs:
			newDoc = []
			for sentence in doc:
				newSentence = []
				# print(f"Sentence: {sentence}")
				wordsInSentence = []
				for word in sentence:
					wordsInSentence.append(word.translate(str.maketrans(string.punctuation, " "*len(string.punctuation))))
				for word in wordsInSentence:
					# if word in string.punctuation:
					# 	continue
					# # print(f"Word under consideration: {word}")
					# wordNetSense = lesk(sentence, word)
					# if wordNetSense:
					# 	# newSentence.extend(wordNetSense.lemma_names())

					# 	newSentence.append(str(wordNetSense).split("'")[1])
					# 	# for holo in wordNetSense.part_holonyms():
					# 	# 	newSentence.append(str(holo).split("'")[1])
					# 	# for entail in wordNetSense.entailments():
					# 	# 	newSentence.append(str(entail).split("'")[1])

					# 	# for ss in wn.synsets(word):
					# 	# 	if ss == wordNetSense:
					# 	# 		defn = list(set([lemmatizer.lemmatize(w) for w in str(ss.definition()).split(" ") if w not in list(stopwords.words('english'))]))
					# 	# 		# print(f"Definition: {defn}")
					# 	# 		newSentence.extend(defn)
					# 	# 		break
					# else:
					newSentence.append(word)
				newDoc.append(newSentence)
			newDocs.append(newDoc)

		docs = newDocs.copy()
		del newDocs

		#Create a list of lists where list stands for the docs corpus
		#and sub-lists are the flattened doc lists in docs array
		newDocs = []
		for doc in docs:
			newDocs.append(list(chain.from_iterable(doc)))

		model = FastText(newDocs, min_count=5)

		wordEmbeddings = {}
		wordsMap = []
		embsMap = []
		allDocEmbs = []

		for doc in newDocs:
			docEmb = []
			for word in doc:
				try:
					emb = model.wv[word]
					docEmb.append(emb)
					wordEmbeddings.update({word: emb})
				except: 
					continue
			allDocEmbs.append(docEmb)
		for word in wordEmbeddings.keys():
			wordsMap.append(word)
			embsMap.append(wordEmbeddings[word])

		labels , centers = spherical_kmeans(K_clusters , embsMap, sim_digree)
		keywords = proportion_keywords(centers, labels, index2word=wordsMap)

		conceptMap = {}
		docConceptCounts = []	
		
		for doc in newDocs:
			clusterCount = [0]*len(keywords)
			f = False
			for word in list(set(doc)):
				for cl in range(len(keywords)):
					if word in [f for (f,s) in keywords[cl]]:
						if str(cl) not in conceptMap.keys():
							conceptMap.update({
								str(cl) : {
									"docFreq": 0,
									"docCounts": [0]*len(newDocs)
								}
							})

						if conceptMap[str(cl)]["docCounts"][newDocs.index(doc)] == 0:
							conceptMap[str(cl)]["docFreq"] += 1
						conceptMap[str(cl)]["docCounts"][newDocs.index(doc)] += 1
						clusterCount[cl] += doc.count(word)
						f = True 
			docConceptCounts.append(clusterCount)
			if not f: 
				# print(f"Not in any cluster: {doc}\n")
				continue
		# for concept in conceptMap.keys():
		# 	df = conceptMap[concept].get("docFreq")
		# 	print(f"{concept}: {df}/{len(newDocs)}")

		docConcept = []

		for doc in newDocs: 
			docCounts = docConceptCounts[newDocs.index(doc)]
			docRepr = [0]*len(docCounts)
			for concept in list(conceptMap.keys()):
				curConcept = docCounts[int(concept)]
				if sum(docCounts):
					curConcept /= sum(docCounts)
				if conceptMap[concept]["docFreq"] != 0:
					curConcept *= np.log(len(newDocs)/conceptMap[concept]["docFreq"])
				docRepr[int(concept)] = curConcept
			docConcept.append(docRepr)

		# fig = visualize_pairwise_distance(centers, max_dist=.7, sort=True)
		# fig.show()

		# words = list(set(chain.from_iterable(list(chain.from_iterable(docs)))))

		# # bigrams = []
		# # for doc in docs:
		# # 	bigramInDoc = list(nltk.bigrams(list(chain.from_iterable(doc))))
		# # 	bigrams.extend(bigramInDoc)
		# # bigrams = list(set(bigrams))

		# documentIndex = {}
		# invertedIndex = {}
		# docsRepr = {}

		# for word in words:
		# 	invertedIndex[word] = {"numOccurs":[], "docFreq":0, "containingDocs":[]}

		# # #Initialise inverted index for bigrams: number of documents the bigram occurs in 
		# # #and list of docIDs the bigram occurs in
		# # for bigr in bigrams:
		# # 	invertedIndex[bigr] = {"docFreq":0, "containingDocs":[]}

		# #Populate the inverted index for words intialised above
		# #Go over each document and incement the docCounts for each word found in that document
		# #Also attack the corresponding docIDs to the list associated with those words
		# for i in range(len(docs)):
		# 	#Flatten the document which is stored as a list of list of words in each sentence 
		# 	#and store the bigrams 
		# 	documentIndex[docIDs[i]] = list(chain.from_iterable(docs[i]))
		# 	#Find top 5 words in document based on frequency
		# 	wordsInDoc = [w for w in documentIndex[docIDs[i]] if w != " "]
		# 	topWords = [w for (w, c) in Counter(wordsInDoc).most_common(min(5, len(wordsInDoc)))]
		# 	for word in list(set(documentIndex[docIDs[i]])):
		# 		if word in words:
		# 			if word in topWords:
		# 				invertedIndex[word]["numOccurs"].append(documentIndex[docIDs[i]].count(word)+10)
		# 			else:
		# 				invertedIndex[word]["numOccurs"].append(documentIndex[docIDs[i]].count(word))
		# 			invertedIndex[word]["docFreq"] += 1
		# 			invertedIndex[word]["containingDocs"].append(docIDs[i])
		# 	# documentIndex[docIDs[i]] = list(nltk.bigrams(list(chain.from_iterable(docs[i]))))
		# 	# for bigr in list(set(documentIndex[docIDs[i]])):
		# 	# 	invertedIndex[bigr]["docFreq"] += 1
		# 	# 	invertedIndex[bigr]["containingDocs"].append(docIDs[i])
		

		# # DFs = []
		# # for bigr in list(invertedIndex.keys()):
		# # 	DFs.append(invertedIndex[bigr]["docFreq"])
		# # # DFs = np.sort(list(set(DFs)))
		# # # print(DFs[::-1])
		# # print(np.mean(DFs))
		# # exit(0)

		# # #Remove bigrams which appear in too less documents
		# # toRem = []
		# # for bigr in list(invertedIndex.keys()):
		# # 	if invertedIndex[bigr]["docFreq"] < 2:
		# # 		if random.uniform(0,1) <= 0.5:
		# # 			toRem.append(bigr) 

		# # for tR in toRem:
		# # 	del invertedIndex[tR]
		# # 	bigrams.remove(tR)

		# #Average document length
		# avdl = 0
		# for doc in docs:
		# 	avdl += len(list(chain.from_iterable(doc)))
		# avdl /= len(docs)
		
		# #Better to precompute document representation because time saving >>> storage 
		# for i in range(len(docs)):
		# 	docRepr = [0]*len(words)
		# 	curDocLen = len(list(chain.from_iterable(docs[i])))
		# 	for word in list(set(documentIndex[docIDs[i]])):
		# 		if word not in words:
		# 			continue
		# 		#TF-IDF weighting of terms
		# 		# termFreq = documentIndex[docIDs[i]].count(word)/len(documentIndex[docIDs[i]])
		# 		# docFreq = np.log(len(docs)/invertedIndex[word]["docFreq"])

		# 		fij = documentIndex[docIDs[i]].count(word)/len(documentIndex[docIDs[i]])
		# 		#Penalise long documents because usually shorter documents are more information dense
		# 		#Here kHP and bHP are hyperparameters
		# 		termFreq = ((k1HP+1)*fij)/(k1HP*(1-b1HP+b1HP*(curDocLen/avdl))+fij)
		# 		docFreq = np.log((len(docs)-invertedIndex[word]["docFreq"]+0.5)/(invertedIndex[word]["docFreq"]+0.5))

		# 		#Logarithmic weighting of terms
		# 		# termFreq = np.log(1+(documentIndex[docIDs[i]].count(word)/len(documentIndex[docIDs[i]])))
		# 		# docFreq = 0
		# 		# for z in range(len(invertedIndex[word]["numOccurs"])):
		# 		# 	docFreq += invertedIndex[word]["numOccurs"][z]*np.log(invertedIndex[word]["numOccurs"][z])
		# 		# docFreq /= np.log(len(docs))
		# 		# docFreq += 1

		# 		docRepr[words.index(word)] = termFreq*docFreq 
		# 	docsRepr[docIDs[i]] = docRepr
		# 	# docRepr = [0]*len(bigrams)
		# 	# for bigr in list(set(documentIndex[docIDs[i]])):
		# 	# 	if bigr not in bigrams:
		# 	# 		continue
		# 	# 	termFreq = documentIndex[docIDs[i]].count(bigr)/len(documentIndex[docIDs[i]])
		# 	# 	docFreq = np.log(len(docs)/invertedIndex[bigr]["docFreq"])
		# 	# 	docRepr[bigrams.index(bigr)] = termFreq*docFreq 
		# 	# docsRepr[docIDs[i]] = docRepr


		# docsTerms = list(docsRepr.values())
		# termsDocs = np.transpose(docsTerms).tolist()
		# #T - term matrix in terms of latent concepts
		# #S - singular values associated with each latent concept
		# #Dh - transpose of the document matrix in terms of latent concepts
		# T, S, Dh = np.linalg.svd(termsDocs)

		# print(S.shape)
		
		# #k-rank approximation
		# t = T[:, 0:self.k].copy()
		# s = S[0:self.k].copy()
		# dh = Dh[0:self.k,:].copy()


		# #Normalise document representations to nullify effect of lengths of documents
		# for j in range(len(dh[0])):
		# 	if np.linalg.norm(dh[:,j]) != 0:
		# 		dh[:,j] /= np.linalg.norm(dh[:,j])
		

		index = {
			# "bigrams": bigrams,
			# "words": words,
			# "documentIndex": documentIndex,
			# "invertedIndex": invertedIndex,
			# "docsRepr": docsRepr,
			# "termsDocs": termsDocs,
			# "t": t,
			# "s": s,
			# "dh": dh,
			"docIDs": docIDs,
			# "avdl": avdl,
			"keywords": keywords,
			"docConceptCounts": docConceptCounts,
			"conceptMap": conceptMap,
			"docConcept": docConcept,
			"wordEmbeddings": wordEmbeddings,
			"centers": centers,
			"labels": labels,
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

		newQueries = list(chain.from_iterable(queries))
		for query in newQueries:
			clusterCount = [0]*len(self.index["keywords"])
			f = False 
			for word in list(set(query)):
				if word not in self.index["wordEmbeddings"].keys():
					continue
				wordEmbed = self.index["wordEmbeddings"][word]
				for c in range(len(self.index["centers"])):
					cent = self.index["centers"][c]
					curCosSim = findCosSim(cent, wordEmbed)
					if curCosSim >= sim_digree:
						clusterCount[int(self.index["labels"][c])] += 1

			for cl in range(len(clusterCount)):
				if sum(clusterCount):
					clusterCount[cl] /= sum(clusterCount)
				if self.index["conceptMap"][str(cl)]["docFreq"] != 0:
					clusterCount[cl] *= np.log(len(self.index["docConceptCounts"])/self.index["conceptMap"][str(cl)]["docFreq"])
			#Calculate cosine similarities
			cosSims = []
			cosSimsIndex = []
			for d in range(len(self.index["docConcept"])):
				cosSim = np.dot(self.index["docConcept"][d],clusterCount)
				cosSims.append(cosSim)
				cosSimsIndex.append(self.index["docIDs"][d])
			doc_IDs_ordered.append([id for _, id in sorted(zip(cosSims, cosSimsIndex),reverse=True)])	
	

		# #Fill in code here
		# for query in queries:
		# 	# bigramsInQuery = list(nltk.bigrams(list(chain.from_iterable(query))))

		# 	lemmatizer = WordNetLemmatizer()
		# 	newQuery = []
		# 	for sentence in query:
		# 		newSentence = []
		# 		wordsInSentence = []
		# 		for word in sentence:
		# 			wordsInSentence.append(word.translate(str.maketrans(string.punctuation, " "*len(string.punctuation))))
		# 		for word in wordsInSentence:
		# 			if word in string.punctuation:
		# 				continue
		# 			wordNetSense = lesk(sentence, word)
		# 			if wordNetSense:
		# 				# newSentence.extend(wordNetSense.lemma_names())
		# 				newSentence.append(str(wordNetSense).split("'")[1])

		# 				for holo in wordNetSense.part_holonyms():
		# 					newSentence.append(str(holo).split("'")[1])
		# 				for entail in wordNetSense.entailments():
		# 					newSentence.append(str(entail).split("'")[1])

		# 				# for ss in wn.synsets(word):
		# 				# 	if ss == wordNetSense:
		# 				# 		defn = list(set([lemmatizer.lemmatize(w) for w in str(ss.definition()).split(" ") if w not in list(stopwords.words('english'))]))
		# 				# 		newSentence.extend(defn)
		# 				# 		break
		# 			else:
		# 				newSentence.append(word)
		# 		newQuery.append(newSentence)

		# 	query = newQuery.copy()
		# 	del newQuery

		# 	wordsInQuery = list(chain.from_iterable(query))

		# 	queryRepr = [0]*len(list(self.index["invertedIndex"].keys()))
		# 	for word in list(set(wordsInQuery)):
		# 	# for bigram in list(set(bigramsInQuery)):
		# 		if word not in list(self.index["invertedIndex"].keys()):
		# 			continue
		# 		# if bigram not in list(self.index["invertedIndex"].keys()):
		# 			# continue

		# 		#TF-IDF weighting of terms
		# 		# termFreq = (wordsInQuery.count(word))/(len(wordsInQuery))
		# 		# docFreq = np.log(len(list(self.index["documentIndex"].keys()))/self.index["invertedIndex"][word]["docFreq"])

		# 		fij = (wordsInQuery.count(word))/(len(wordsInQuery))
		# 		#Here k2HP and b2HP are hyperparameters
		# 		termFreq = ((k2HP + 1)*fij)/(k2HP + fij)
		# 		docFreq = np.log((len(list(self.index["documentIndex"].keys()))-self.index["invertedIndex"][word]["docFreq"]+0.5)/(self.index["invertedIndex"][word]["docFreq"]+0.5))
				
		# 		#Logarithmic weighting of terms
		# 		# termFreq = np.log(1+(wordsInQuery.count(word))/(len(wordsInQuery)))
		# 		# docFreq = 0
		# 		# for z in range(len(self.index["invertedIndex"][word]["numOccurs"])):
		# 		# 	docFreq += self.index["invertedIndex"][word]["numOccurs"][z]*np.log(self.index["invertedIndex"][word]["numOccurs"][z])
		# 		# docFreq /= np.log(len(list(self.index["documentIndex"].keys())))
		# 		# docFreq += 1
				
		# 		queryRepr[list(self.index["invertedIndex"].keys()).index(word)] = termFreq*docFreq
		# 		# termFreq = (bigramsInQuery.count(bigram))/(len(bigramsInQuery))
		# 		# docFreq = np.log(len(list(self.index["documentIndex"].keys()))/self.index["invertedIndex"][bigram]["docFreq"])
		# 		# queryRepr[list(self.index["invertedIndex"].keys()).index(bigram)] = termFreq*docFreq

		# 	pseudoDoc = np.dot(queryRepr, np.dot(self.index["t"], np.linalg.inv(np.diag(self.index["s"]))))
		# 	if np.linalg.norm(pseudoDoc) != 0:
		# 		pseudoDoc /= np.linalg.norm(pseudoDoc)

		# 	cosSims = []
		# 	cosSimsIndex = []
		# 	for d in range(len(self.index["dh"].T)):
		# 		cosSim = np.dot(self.index["dh"].T[d],pseudoDoc)
		# 		# if cosSim < 0.5:
		# 		# 	continue
		# 		cosSims.append(cosSim)
		# 		cosSimsIndex.append(self.index["docIDs"][d])
		# 	doc_IDs_ordered.append([id for _, id in sorted(zip(cosSims, cosSimsIndex),reverse=True)])	

		# 	# cosSims = np.dot(self.index["dh"].T,pseudoDoc)
		# 	# cosSimsSort = np.argsort(cosSims)

		# 	# cur_doc_IDs_ordered = [0]*len(self.index["docIDs"])
		# 	# for i in range(len(cosSimsSort)):
		# 	# 	cur_doc_IDs_ordered[cosSimsSort[i]] = self.index["docIDs"][i]
		# 	# doc_IDs_ordered.append(cur_doc_IDs_ordered)

		# 	# print(cosSims)
		# 	# cosSims = np.argpartition(cosSims, -len(cosSims))[-len(cosSims):]
		# 	# print(cosSims)
		# 	# exit(0)
		# 	# # doc_IDs_ordered.append(cosSims)
		# 	# # doc_IDs_ordered.append(np.array(self.index["docIDs"])[cosSims])
		# 	# # print("1)")
		# 	# # print(doc_IDs_ordered)
		# 	# # doc_IDs_ordered = []

		# 	# docsRelToQuery = list(set(docsRelToQuery))
		# 	# docsRelToQueryRepr = []
		# 	# for (ind, doc) in enumerate(docsRelToQuery):
		# 	# 	# docRepr = [0]*len(self.index["invertedIndex"])
		# 	# 	# for (wIndex, word) in enumerate(self.index["documentIndex"][doc]):
		# 	# 	# 	termFreq = self.index["documentIndex"][doc].count(word)/len(self.index["documentIndex"][doc])
		# 	# 	# 	docFreq = np.log(len(list(self.index["documentIndex"].keys()))/self.index["invertedIndex"][word]["docFreq"])
		# 	# 	# 	docRepr[ind] = termFreq*docFreq
		# 	# 	# docRepr /= np.linalg.norm(docRepr)
		# 	# 	docRepr = self.index["docsRepr"][doc]
		# 	# 	docsRelToQueryRepr.append(docRepr)

		# 	# cosSims = np.dot(queryRepr, np.transpose(docsRelToQueryRepr))
		# 	# doc_IDs_ordered.append([docsRelToQuery[idPos] for idPos in np.argsort(cosSims)[::-1]])
		# 	## doc_IDs_ordered.append([list(self.index["docsRepr"].keys())[rank] for rank in np.argsort(cosSims)[::-1]])
		return doc_IDs_ordered




