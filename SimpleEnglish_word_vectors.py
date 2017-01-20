# coding: utf-8

import logging, re, os, bz2, pickle
from collections import Counter
from operator import itemgetter
import itertools

# Example from one subtitle of the OpenSubtitles2016 dataset. Guess the movie?
# 
# +--------------+----------------+---------------+------------+------------+-----------+-----------+------------+
# | Topic 1      | Topic 2        | Topic 3       | Topic 4    | Topic 5    | Topic 6   | Topic 7   | Topic 8    |
# |--------------+----------------+---------------+------------+------------+-----------+-----------+------------|
# | dinosaur     | soft_tissue    | raptors       | interest   | attraction | prevent   | teeth     | bones      |
# | jurassic     | dna            | free_radicals | existence  | motion     | determine | turn      | rides      |
# | stegosaurus  | oil            | lawyers       | experience | ammunition | coverage  | blood     | caves      |
# | dinosaurs    | cytosine       | costs         | reality    | situation  | withstand | ride      | drones     |
# | species      | cell_membranes | incidents     | figure     | extraction | phase     | shoulder  | shirts     |
# | extinct      | aluminum       | assets        | principles | fraction   | pre       | room      | boots      |
# | ankylosaurus | predator       | excavators    | fence      | innovation | evaluate  | bunch     | walls      |
# |              | venom          | others        | focus      | protection | tense     | glass     | breeds     |
# |              | proteins       | visitors      | concern    | sense      | captivity | breaks    | woods      |
# |              | tunnels        | profits       | respect    | solution   | terminate | breach    | tree_frogs |
# +--------------+----------------+---------------+------------+------------+-----------+-----------+------------+

# Required data science packages
# ------------------------------
import gensim

## For scraping text out of a wikipedia dump. Get dumps at https://dumps.wikimedia.org/backup-index.html
from gensim.corpora import WikiCorpus

## Latest greatest word vectors. See https://github.com/facebookresearch/fastText (code also available on pypi).
## Earlier code and literature is google word2vec (same guys, moved from google to facebook). 
import fasttext

## For computing phrases from input text. 
## The word2vec literature and c implementation is at https://github.com/tmikolov/word2vec. 
## The gensim python implementation is used below. 
from gensim.models.phrases import Phrases

## Use nltk.punkt to segment text into sentences.
import nltk

## Try to keep raw input data separate from computed models. These models will be used as
## input to follow on vec2topic code.
data_directory = 'data/'
model_directory = 'models/'

## Setup for logging.
from imp import reload
reload(logging)

LOG_FILENAME = data_directory + 'vec2topic.log'
#logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s',"%b-%d-%Y %H:%M:%S")
logger.handlers[0].setFormatter(formatter)

## Main inputs to program. Data for model and name of knowledge base (background language model).
knowledge_base = 'simplewiki-20160820-pages-articles.xml.bz2'
knowledge_base_prefix = 'simplewiki-20160820-pages-articles'

# Word vector dimension for knowledge base.
knowledge_base_vector_dimension = 300    

## Intermediate files generated from inputs.
## First is the output from scraping text out of the wikipedia dump.
knowledge_base_text = data_directory + knowledge_base_prefix + '.txt'

## To make more meaningful 
knowledge_base_bigrams = model_directory + knowledge_base_prefix + '_bigrams.pkl'
knowledge_base_phrases = data_directory + knowledge_base_prefix + '_phrases.txt'

nowledge_base_model = model_directory + knowledge_base_prefix + '.bin'
knowledge_base_vectors = model_directory + knowledge_base_prefix + '.vec'


# Global knowledge vectors -- English Wikipedia
# --------------------------------------------
# First step is to compute word embeddings of a global knowledge base from the English Wikipedia to capture
# the generic meaning of words in widely used contexts.
# 
# The gensim package has examples of processing wikipedia dumps as well as streaming corpus implementation. 

# ### Process wikipedia dump
# First download the wikipedia dump and place it in the data directory before running this notebook. The cell 
# below will use the gensim class WikiCorpus to strip the wikipedia markup and store each article as one line 
# of the output text file. Only do these computations once if possible.

# if not os.path.isfile(knowledge_base_text):
#     space = ' '
#     i = 0
#     output = open(knowledge_base_text, 'wb# ')
#     logger.info('Processing knowledge base %s', knowledge_base)
#     wiki = WikiCorpus(data_directory + knowledge_base, lemmatize=False, dictionary={})
#     for text in wiki.get_texts():
#         output.write(space.join(text) + "\n")
#         i = i + 1
#         if (i % 10000 == 0):
#             logger.info("Saved " + str(i) + " articles")
#     output.close()
#     logger.info("Finished Saved " + str(i) + " articles")
# else:
#     logger.info('Knowledge base %s already on disk.', knowledge_base_text)

### Extract Phrases ###
## See http://arxiv.org/abs/1310.4546 (section 4) for a discription of algorithm using unigram and bigram counts.
## Could compute word vectors directly from the text, but then miss interesting compound words. For some languages,
## instead rely on natural language proccessing code to segment text into words (including compounds).

## May be overkill to extract sentences in the phrase detection. 

def read_sents_from_data(path):
    with bz2.BZ2File(path, 'rb') as data:    
        for title, text, pageid in gensim.corpora.wikicorpus.extract_pages(data):
            text = gensim.corpora.wikicorpus.filter_wiki(text)
            sents = nltk.sent_tokenize(text.lower())
            for sent in sents:
                yield nltk.word_tokenize(sent)

## Generating phrases is expensive, save them as a pickle file.
if not os.path.isfile(knowledge_base_bigrams):
    logger.info('Generate and save copy of knowledge base bigrams %s', knowledge_base_bigrams)
    kb_bigrams = Phrases(read_sents_from_data(data_directory + knowledge_base), threshold=100.0)
    with open(knowledge_base_bigrams,'w') as bigrams_fp:
        pickle.dump(kb_bigrams, bigrams_fp)
else:
    logger.info('Read copy of knowledge base bigrams %s', knowledge_base_bigrams)
    with open(knowledge_base_bigrams,'r') as bigrams_fp:
        kb_bigrams = pickle.load(bigrams_fp)   

## In practice, need to set a threshhold for the phrase detection. To find a good threshold, look at the 
## generated phrases for a given value.
count = 0
for phrase, score in kb_bigrams.export_phrases(read_sents_from_data(data_directory + knowledge_base)):
    if True and count >= 1000:
        break
    print(u'{0}\t{1}'.format(phrase.decode('utf-8'), score))
    count += 1


if not os.path.isfile(knowledge_base_phrases):
    with open(knowledge_base_phrases, 'w') as data:
        for sent in read_sents_from_data(data_directory + knowledge_base):
            s = ' '.join(kb_bigrams[sent]) + u'\n'
            data.write(s.encode('utf-8'))
    logger.info('Saved copy of knowledge base phrases %s', knowledge_base_phrases)
else:    
    logger.info('Copy of knowledge base phrases %s on disk.', knowledge_base_phrases)

# ### Compute word vectors for knowledge base

# Some computational performances comparing `word2vec` vs. `fasttext`. 
# 
# For computing full wikipedia using `word2vec`, using 300 dimensional word vectors, need to filter vocabulary 
# so that basic memory usage of word2vec fits in physical memory. 
# 
# > the `syn0` structure holding (input) word-vectors-in-training will require:
# > 5759121 (your vocab size) * 600 (dimensions) * 4 bytes/dimension = 13.8GB
# > The `syn1neg` array (hidden->output weights) will require another 13.8GB.
#
# min_count = 10 results in 2,947,700 words (requires more than 7G physical memory)
# min_count = 5 results in 4,733,171 words (requires more than 11G physical memory)
# min_count = 0 results in 11,631,317 words (requires more than 28G physical memory)
#
# 
# Using `fasttext` with `min_count=5`, `bucket=2000000`, and `t=1e-4` on enwiki, used a constant 8.43G memory 
# used during computation (over 10 hours 8-core, 16G ram). Final vocabulary has 2,114,311 words.

if not os.path.isfile(knowledge_base_vectors):
    knowledge_base_skipgram = fasttext.skipgram(knowledge_base_phrases, 
        model_directory + knowledge_base_prefix, lr=0.02, 
        dim=knowledge_base_vector_dimension, ws=5, word_ngrams=1,
        epoch=1, min_count=5, neg=5, loss='ns', bucket=2000000, minn=3, maxn=6,
        thread=8, t=1e-4, lr_update_rate=100)
else:
    logger.info('Knowledge vectors %s already on disk.', knowledge_base_vectors)
    knowledge_base_skipgram = fasttext.load_model(knowledge_base_model)


