from util import *

# Add your import statements here

from nltk.corpus import stopwords


class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []

		#Fill in code here

		nltk_stopwords = stopwords.words("english")

		for i in text:
			sub_list = []
			for j in i:
				if j not in nltk_stopwords:
					sub_list.append(j)
			stopwordRemovedText.append(sub_list)

		return stopwordRemovedText




	