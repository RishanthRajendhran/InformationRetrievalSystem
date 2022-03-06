from util import *

# Add your import statements here
from nltk.tokenize import TreebankWordTokenizer
  


class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here
		for sentence in text:
			# Splits at space 
			tokenizedText.append(sentence.split()) 

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []
		tokenizer = TreebankWordTokenizer()
		for sentence in text:
			# Splitting using penn tree bank tokenizer 
			tokenizedText.append(tokenizer.tokenize(sentence))
		
		#Fill in code here

		return tokenizedText