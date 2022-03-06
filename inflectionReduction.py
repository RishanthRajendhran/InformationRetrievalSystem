from util import *

# Add your import statements here

from nltk.stem import PorterStemmer 


class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = None

		#Fill in code here
		
		wns = PorterStemmer()

		for i in range(len(text)):
			for j in range(len(text[i])):
				text[i][j] = wns.stem(text[i][j])

		reducedText = text
		return reducedText


