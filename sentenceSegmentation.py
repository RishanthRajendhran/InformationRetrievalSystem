from util import *

# Add your import statements here
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")



class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		segmentedText = [s+"." if not ((len(s) >= 2 and s[-2]=="." and s[-1] == " ") or (len(s)>=1 and s[-1] == ".")) else s for s in text.split(". ")]

		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		segmentedText = sent_tokenize(text)
		
		return segmentedText