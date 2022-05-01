# Add your import statements here
import nltk 
nltk.download("stopwords")

nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np

import random

import string

from collections import Counter

import gensim
from gensim.models import FastText
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import wikipedia

# Add any utility functions here