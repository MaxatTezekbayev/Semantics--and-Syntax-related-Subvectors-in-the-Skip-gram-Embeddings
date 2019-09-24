# Semantics--and-Syntax-related-Subvectors-in-the-Skip-gram-Embeddings

A Python implementation of the empirical part of the paper:\
**"Semantics- and Syntax-related Subvectors in the Skip-gram Embeddings”** \
*Maxat Tezekbayev, Zhenisbek Assylbekov, Rustem Takhanov* 

# Contacts
**Authors**: Maxat Tezekbayev, Zhenisbek Assylbekov, Rustem Takhanov\
**Pull requests and issues**: maxat.tezekbayev@nu.edu.kz, maksat013@gmail.com

# Contents
We show that the skip-gram embedding of any word can be decomposed into two subvectors which roughly correspond to semantic and syntactic roles of the word.

**Keywords**: natural language processing, words embeddings

We trained SGNS with tied weights (Assylbekov and Takhanov 2019) on two widely-used datasets,text8 and enwik9 which gives us word embeddings as well as their partitions
<img src="https://latex.codecogs.com/gif.latex?R^{$$\mathbf{w}_i^\top:=[\mathbf{x}_i^\top;\mathbf{y}_i^\top].$$}" 
We used the reference word2vec implementation from the TensorFlow codebase with all hyperparameters set to their default values except that we choose the learning rate to decay 20% faster in the weight-tied model. (https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py). 





standard nowadays the trained embeddings
are evaluated on several word similarity and word analogy tasks. We used the hyperwords
tool of Goldberg and Levy (2014) and we refer the reader to their paper for the methodology
of evaluation.



We performed such comparison on datasets from the SemEval Semantic Textual Similarity (STS) tasks (http://ixa2.si.ehu.es/stswiki/index.php/Main_Page, test datasets) with GLOVE and Word2Vec word embeddings:

a.	Glove and Word2Vec word vectors were trained on the same dataset (Enwik 9), with the same set up (min count = 50, dimension of the word vector = 200). (Code for the customized training of the word models: /data/Training Word2Vec model with custom set up.ipynb). 

b.	Pre-trained GLOVE word vectors (Common Crawl, 840B tokens, 2.2M vocab, cased, 300d vectors) (can be downloaded from https://nlp.stanford.edu/projects/glove/). \
	Word2Vec vectors trained on Enwik 9, with min_count = 50, window size =2, vector dimension = 300. (Code for the customized training of the word models: /data/Training Word2Vec model with custom set up.ipynb)

Data also contains txt file with words and its frequencies (enwiki_vocab_min200.txt), which is used in SIF model implementation. 
