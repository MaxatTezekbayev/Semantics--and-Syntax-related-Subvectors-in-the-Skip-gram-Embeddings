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

1. We trained SGNS with tied weights (Assylbekov and Takhanov 2019) on two widely-used datasets,text8 and enwik9 (preprocessed with wikifil.pb) which gives us word embeddings as well as their partitions. We used the reference word2vec implementation from the TensorFlow codebase with all hyperparameters set to their default values except that we choose the learning rate to decay 20% faster in the weight-tied model. (https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py). 

The train can be done via:
```
python3 word2vec_tied.py -train_data text8 -train True -gen_embs False  -postag	False
```

However, before train, you will need to compile the ops as follows:
```
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L$TF_LIB -ltensorflow_framework
```


2. We extracted the whole vectors **w**’s, as well as the subvectors **x**’s and **y**’s to txt files. In order to extract embeddings from the trained model you will need to change model.ckpt number in **savedmodel_paths** variable in ***word2vec_tied.py**
```
python3 word2vec_tied.py -train_data text8 -gen_embs True  -postag False
```

3.  We used the HYPERWORDS tool of Levy, Goldberg, and Dagan (https://bitbucket.org/omerlevy/hyperwords/src/default/) in order to evaluate embeddings on standard semantic tasks — word similarity and word analogy.

Commands used for HYPERWORDS can be found in "hyperwords commands.txt"

4. We trained a softmax regression by feeding in the embedding of a current word to predict the part-of-speech (POS) tag of the next word. We evaluated the whole vectors and the subvectors on tagging  the  Brown  corpus  with  the  Universal  POS  tags. Words which were not in our Word2vec model were excluded from dataset and last 20% of the words were used as test dataset.
```
python3 word2vec_tied.py -train_data text8 -gen_embs False -postag True
```

5. For calculating dot products and cosine similarities between embedding of given word (for example **dog**) and embeddings of other words (as well as for **w** and **x**,**y**).
```
python3 word2vec_tied.py -train_data text8 -near_words_to dog
```
