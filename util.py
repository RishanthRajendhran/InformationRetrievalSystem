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
from soyclustering import SphericalKMeans
from scipy.sparse import csr_matrix
from soyclustering import proportion_keywords
from soyclustering import visualize_pairwise_distance

# Add any utility functions here