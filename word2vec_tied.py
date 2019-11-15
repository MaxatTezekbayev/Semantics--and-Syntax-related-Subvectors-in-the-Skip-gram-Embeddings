from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#fil9 is preprocessed enwik9
save_paths={'text8':'tmp',
           'fil9':'tmp_fil9'}

embedding_folders={'text8':'embeddings',
                   'fil9':'embeddings_fil9'}



savedmodel_paths={'text8':"tmp/model.ckpt-70508708",
                   'fil9':"tmp_fil9/model.ckpt-463085955"}


import tensorflow as tf
import os
import numpy as np
import pandas as pd
rng = np.random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.20
config.gpu_options.allow_growth = True #allocate dynamically 
sess = tf.Session(config = config)
import os
import sys
import threading
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

tf.reset_default_graph()

word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath('__file__')), 'word2vec_ops.so'))

flags = tf.app.flags
#our flags
flags.DEFINE_string("train", "False",'Flag for train')
flags.DEFINE_string("gen_embs", "True",'Flag for generating txt files with embeddings')
flags.DEFINE_string("postag", "True",'Flag for training pos tagger')
flags.DEFINE_string("near_words_to",None,'Flag for word to see nearest words around')
flags.DEFINE_string("continue_train","False",'Flag continue train of given saved model')

#standard flags
flags.DEFINE_string("train_data", 'text8', "Training text file. "
                    "E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
dataset=flags.FLAGS.train_data
flags.DEFINE_string("save_path", save_paths[dataset], "Directory to write the model and "
                    "training summaries.")

flags.DEFINE_string(
    "eval_data", "questions-words.txt", "File consisting of analogies of four tokens."
    "embedding 2 - embedding 1 + embedding 3 should be close "
    "to embedding 4."
    "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 100,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 16,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")
flags.DEFINE_integer("statistics_interval", 100,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 1800,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 1800,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval).")

FLAGS = flags.FLAGS





class Options(object):
    """Options used by our word2vec model."""
    def __init__(self):
    # Model options.

        # Embedding dimension.
        self.emb_dim = FLAGS.embedding_size

        # Training options.
        # The training text file.
        self.train_data = FLAGS.train_data

        # Number of negative samples per example.
        self.num_samples = FLAGS.num_neg_samples

        # The initial learning rate.
        self.learning_rate = FLAGS.learning_rate

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.epochs_to_train = FLAGS.epochs_to_train

        # Concurrent training steps.
        self.concurrent_steps = FLAGS.concurrent_steps

        # Number of examples for one training step.
        self.batch_size = FLAGS.batch_size

        # The number of words to predict to the left and right of the target word.
        self.window_size = FLAGS.window_size

        # The minimum number of word occurrences for it to be included in the
        # vocabulary.
        self.min_count = FLAGS.min_count

        # Subsampling threshold for word occurrence.
        self.subsample = FLAGS.subsample

        # How often to print statistics.
        self.statistics_interval = FLAGS.statistics_interval

        # How often to write to the summary file (rounds up to the nearest
        # statistics_interval).
        self.summary_interval = FLAGS.summary_interval

        # How often to write checkpoints (rounds up to the nearest statistics
        # interval).
        self.checkpoint_interval = FLAGS.checkpoint_interval

        # Where to write out summaries.
        self.save_path = FLAGS.save_path
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)

        # Eval options.
        # The text file for eval.
        self.eval_data = FLAGS.eval_data

class Word2Vec(object):
    """Word2Vec model (Skipgram)."""
    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        self.build_graph()
        self.build_eval_graph()
        self.save_vocab()
    
    def read_analogies(self):
        """Reads through the analogy question file.
        Returns:
          questions: a [n, 4] numpy array containing the analogy question's
                     word ids.
          questions_skipped: questions skipped due to unknown words.
          """
        questions = []
        questions_skipped = 0
        with open(self._options.eval_data, "rb") as analogy_f:
            for line in analogy_f:
                if line.startswith(b":"):  # Skip comments.
                    continue
            words = line.strip().lower().split(b" ")
            ids = [self._word2id.get(w.strip()) for w in words]
            if None in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(np.array(ids))
        print("Eval analogy file: ", self._options.eval_data)
        print("Questions: ", len(questions))
        print("Skipped: ", questions_skipped)
        self._analogy_questions = np.array(questions, dtype=np.int32)

    def forward(self, examples, labels):
        """Build the graph for the forward pass."""
        opts = self._options

        # Declare all variables we need.
        # Embedding: [vocab_size, emb_dim]
        init_width = 0.5 / opts.emb_dim
        emb = tf.Variable(
            tf.random_uniform(
                [opts.vocab_size, opts.emb_dim], -init_width, init_width),
            name="emb")
        self._emb = emb

        Q_np=np.eye(opts.emb_dim,dtype='float32')
        for i in range(0,opts.emb_dim,2):
            Q_np[i,i]=-1.0
        Q=tf.constant(Q_np)

        # Global step: scalar, i.e., shape [].
        self.global_step = tf.Variable(0, name="global_step")

        # Nodes to compute the nce loss w/ candidate sampling.
        labels_matrix = tf.reshape(
            tf.cast(labels,
                    dtype=tf.int64),
            [opts.batch_size, 1])

        # Negative sampling.
        sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=opts.num_samples,
            unique=True,
            range_max=opts.vocab_size,
            distortion=0.75,
            unigrams=opts.vocab_counts.tolist()))

        # Embeddings for examples: [batch_size, emb_dim]
        example_emb = tf.nn.embedding_lookup(emb, examples)
        labels_emb=tf.nn.embedding_lookup(emb,labels)
        sampled_ids_emb=tf.nn.embedding_lookup(emb,sampled_ids)
        # Weights for labels: [batch_size, emb_dim]
        true_w=tf.matmul(labels_emb,Q)

        # Weights for sampled ids: [num_sampled, emb_dim]
        sampled_w=tf.matmul(sampled_ids_emb,Q)

        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1)

        norm_emb=tf.nn.l2_normalize(emb, 1)
        inp=tf.placeholder(dtype=tf.int32)

        #dot product between word vector and context vector w_i*c_j=w_i*Q*w_j
        output=tf.matmul(tf.nn.embedding_lookup(emb,inp),tf.matmul(emb,Q),transpose_b=True)
        output_score, output_idx = tf.nn.top_k(output, min(100000, self._options.vocab_size))

        

        emb_x=emb[:,1::2]
        emb_y=emb[:,::2]

        #dot product between w_i * w_j
        dot_product=tf.matmul(tf.nn.embedding_lookup(emb,inp),emb,transpose_b=True)
        dot_product_score, dot_product_idx = tf.nn.top_k(dot_product, min(100000, self._options.vocab_size))


        #dot product between x_i * x_j
        dot_product_x=tf.matmul(tf.nn.embedding_lookup(emb_x,inp),emb_x,transpose_b=True)
        dot_product_score_x, dot_product_idx_x= tf.nn.top_k(dot_product_x, min(100000, self._options.vocab_size))

        #dot product between x_i * x_j
        dot_product_y=tf.matmul(tf.nn.embedding_lookup(emb_y,inp),-emb_y,transpose_b=True)
        dot_product_score_y, dot_product_idx_y= tf.nn.top_k(dot_product_y, min(100000, self._options.vocab_size))
        

        self._inp=inp
        self._output=output
        self._output_score=output_score
        self._output_idx=output_idx

        self._dot_product=dot_product
        self._dot_product_score=dot_product_score
        self._dot_product_idx=dot_product_idx

        self._dot_product_x=dot_product_x
        self._dot_product_score_x=dot_product_score_x
        self._dot_product_idx_x=dot_product_idx_x

        self._dot_product_y=dot_product_y
        self._dot_product_score_y=dot_product_score_y
        self._dot_product_idx_y=dot_product_idx_y

        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True)
        return true_logits, sampled_logits

    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / opts.batch_size
        return nce_loss_tensor

    def optimize(self, loss):
        """Build the graph to optimize the loss function."""

        # Optimizer nodes.
        # Linear learning rate decay.
        opts = self._options
        words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
        lr = opts.learning_rate * tf.maximum(
            0.0001, 1.0 - 1.2*tf.cast(self._words, tf.float32) / words_to_train)
        self._lr = lr
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss,
                                   global_step=self.global_step,
                                   gate_gradients=optimizer.GATE_NONE)
        self._train = train

    def build_eval_graph(self):
        """Build the eval graph."""
        # Eval graph

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

        # Normalized word embeddings of shape [vocab_size, emb_dim].
        nemb = tf.nn.l2_normalize(self._emb, 1)

        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.gather(nemb, analogy_a)  # a's embs
        b_emb = tf.gather(nemb, analogy_b)  # b's embs
        c_emb = tf.gather(nemb, analogy_c)  # c's embs

        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)

        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist = tf.matmul(target, nemb, transpose_b=True)

        # For each question (row in dist), find the top 4 words.
        _, pred_idx = tf.nn.top_k(dist, 4)



        #normalization separately for +1 and for -1:
        nemb_1 = tf.nn.l2_normalize(self._emb[:,1::2], 1)
        nemb_neg1 = tf.nn.l2_normalize(self._emb[:,::2], 1)

        # Nodes for computing neighbors for a given word according to
        # their cosine distance.
        nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_emb_1=tf.gather(nemb_1, nearby_word)
        nearby_emb_neg1=tf.gather(nemb_neg1, nearby_word)


        #near words by entire
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, min(10000, self._options.vocab_size))
        #near words by +1-s
        dist_1 = tf.matmul(nearby_emb_1, nemb_1, transpose_b=True)
        nearby_val_1, nearby_idx_1 = tf.nn.top_k(dist_1,min(10000, self._options.vocab_size))
        #near words by -1-s (y)
        dist_neg1= tf.matmul(nearby_emb_neg1, nemb_neg1, transpose_b=True)
        nearby_val_neg1, nearby_idx_neg1 = tf.nn.top_k(dist_neg1,min(10000, self._options.vocab_size))
        
        #far words by -1-s
        far_val_neg1, far_idx_neg1=tf.nn.top_k(-dist_neg1,min(10000, self._options.vocab_size))



        neighbors=tf.placeholder(dtype=tf.int32)
        neighbors_emb = tf.gather(nemb, neighbors)
        neighbors_emb_1 = tf.gather(nemb_1, neighbors)
        neighbors_emb_neg1 = tf.gather(nemb_neg1, neighbors)
        self._neighbors_emb=neighbors_emb
        neighbors_sim=tf.matmul(nearby_emb,neighbors_emb,transpose_b=True)
        neighbors_sim_1=tf.matmul(nearby_emb_1,neighbors_emb_1,transpose_b=True)
        neighbors_sim_neg1=tf.matmul(nearby_emb_neg1,neighbors_emb_neg1,transpose_b=True)



        #king-man+woman and similarity
        nearby_analogy_dist=tf.matmul(target, nemb, transpose_b=True)
        nearby_analogy_val, nearby_analogy_idx = tf.nn.top_k(nearby_analogy_dist, min(1000, self._options.vocab_size))
        far_analogy_val, far_analogy_idx = tf.nn.top_k(-nearby_analogy_dist, min(1000, self._options.vocab_size))


        #king-man+woman and similarity +1-s

        target_plus1=tf.gather(nemb_1, analogy_c)+(tf.gather(nemb_1, analogy_b)-tf.gather(nemb_1, analogy_a))
        nearby_analogy_dist_plus1=tf.matmul(target_plus1, nemb_1, transpose_b=True)
        nearby_analogy_val_plus1, nearby_analogy_idx_plus1 = tf.nn.top_k(nearby_analogy_dist_plus1, min(1000, self._options.vocab_size))

        target_neg1=tf.gather(nemb_neg1, analogy_c)+(tf.gather(nemb_neg1, analogy_b)-tf.gather(nemb_neg1, analogy_a))
        nearby_analogy_dist_neg1=tf.matmul(target_neg1, nemb_neg1, transpose_b=True)
        nearby_analogy_val_neg1, nearby_analogy_idx_neg1 = tf.nn.top_k(nearby_analogy_dist_neg1, min(1000, self._options.vocab_size))


        far_analogy_val_plus1, far_analogy_idx_plus1 = tf.nn.top_k(-nearby_analogy_dist_plus1, min(1000, self._options.vocab_size))
        far_analogy_val_neg1, far_analogy_idx_neg1 = tf.nn.top_k(-nearby_analogy_dist_neg1, min(1000, self._options.vocab_size))


        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx

        #near words given word
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx
        self._nearby_val_1 = nearby_val_1
        self._nearby_idx_1 = nearby_idx_1
        self._nearby_val_neg1 = nearby_val_neg1
        self._nearby_idx_neg1 = nearby_idx_neg1
        self._far_val_neg1= far_val_neg1
        self._far_idx_neg1 = far_idx_neg1
        #calculate similarities of words given word
        self._neighbors=neighbors
        self._neighbors_sim=neighbors_sim
        self._neighbors_sim_1=neighbors_sim_1
        self._neighbors_sim_neg1=neighbors_sim_neg1

        self._nearby_analogy_val = nearby_analogy_val
        self._nearby_analogy_idx = nearby_analogy_idx

        self._far_analogy_val = far_analogy_val
        self._far_analogy_idx = far_analogy_idx



        self._nearby_analogy_val_neg1 = nearby_analogy_val_neg1
        self._nearby_analogy_idx_neg1 = nearby_analogy_idx_neg1




        self._far_analogy_val_neg1 = far_analogy_val_neg1
        self._far_analogy_idx_neg1 = far_analogy_idx_neg1

        self._nearby_analogy_val_plus1 = nearby_analogy_val_plus1
        self._nearby_analogy_idx_plus1 = nearby_analogy_idx_plus1


        self._far_analogy_val_plus1 = far_analogy_val_plus1
        self._far_analogy_idx_plus1 = far_analogy_idx_plus1

        self._nemb_1=nemb_1
        self._nemb=nemb
        self._nemb_neg1=nemb_neg1

    
    
    def build_graph(self):
        """Build the graph for the full model."""
        opts = self._options
        # The training data. A text file.
        (words, counts, words_per_epoch, self._epoch, self._words, examples,
         labels) = word2vec.skipgram_word2vec(filename=opts.train_data,
                                              batch_size=opts.batch_size,
                                              window_size=opts.window_size,
                                              min_count=opts.min_count,
                                              subsample=opts.subsample)
        (opts.vocab_words, opts.vocab_counts,
         opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
        opts.vocab_size = len(opts.vocab_words)
        print("Data file: ", opts.train_data)
        print("Vocab size: ", opts.vocab_size - 1, " + UNK")
        print("Words per epoch: ", opts.words_per_epoch)
        self._examples = examples
        self._labels = labels
        self._id2word = opts.vocab_words
        for i, w in enumerate(self._id2word):
          self._word2id[w] = i
        true_logits, sampled_logits = self.forward(examples, labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        tf.summary.scalar("NCE_loss", loss)
        self._loss = loss
        self.optimize(loss)

        # Properly initialize all variables.
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()

    def save_vocab(self):
        """Save the vocabulary to a file so the model can be reloaded."""
        opts = self._options
        with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
            for i in xrange(opts.vocab_size):
                vocab_word = tf.compat.as_text(opts.vocab_words[i]).encode("utf-8")
                f.write("%s %d\n" % (vocab_word,
                                     opts.vocab_counts[i]))

            
    def _train_thread_body(self):
        initial_epoch, = self._session.run([self._epoch])
        while True:
            _, epoch = self._session.run([self._train, self._epoch])
            if epoch != initial_epoch:
                break

    def train(self):
        """Train the model."""
        opts = self._options

        initial_epoch, initial_words = self._session.run([self._epoch, self._words])

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(opts.save_path, self._session.graph)
        workers = []
        for _ in xrange(opts.concurrent_steps):
            t = threading.Thread(target=self._train_thread_body)
            t.start()
            workers.append(t)

        last_words, last_time, last_summary_time = initial_words, time.time(), 0
        last_checkpoint_time = 0
        while True:
            time.sleep(opts.statistics_interval)  # Reports our progress once a while.
            (epoch, step, loss, words, lr) = self._session.run(
              [self._epoch, self.global_step, self._loss, self._words, self._lr])
            now = time.time()
            last_words, last_time, rate = words, now, (words - last_words) / (
              now - last_time)
            print("Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r" %
                (epoch, step, lr, loss, rate), end="")
            sys.stdout.flush()
            if now - last_summary_time > opts.summary_interval:
                summary_str = self._session.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                last_summary_time = now
            if now - last_checkpoint_time > opts.checkpoint_interval:
                self.saver.save(self._session,
                                os.path.join(opts.save_path, "model.ckpt"),
                                global_step=step.astype(int))
                last_checkpoint_time = now
            if epoch != initial_epoch:
                break

        for t in workers:
            t.join()

        return epoch

    def _predict(self, analogy):
        """Predict the top 4 answers for analogy questions."""
        idx, = self._session.run([self._analogy_pred_idx], {
            self._analogy_a: analogy[:, 0],
            self._analogy_b: analogy[:, 1],
            self._analogy_c: analogy[:, 2]
        })
        return idx
  
    def _predict_output(self,words,num=20):
        ids = np.array([self._word2id.get(x, 0) for x in words])
        output_words=[]
        vals, idx = self._session.run([self._output_score, self._output_idx], {self._inp: ids})
        for i in xrange(len(words)):
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                output_words.append((self._id2word[neighbor].decode("utf-8") , round(distance,3)))
        return output_words


    def eval(self):
        """Evaluate analogy questions and reports accuracy."""

        # How many questions we get right at precision@1.
        correct = 0

        try:
            total = self._analogy_questions.shape[0]
        except AttributeError as e:
            raise AttributeError("Need to read analogy questions.")

        start = 0
        while start < total:
            limit = start + 2500
            sub = self._analogy_questions[start:limit, :]
            idx = self._predict(sub)
            start = limit
            for question in xrange(sub.shape[0]):
                for j in xrange(4):
                    if idx[question, j] == sub[question, 3]:
                # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
                        correct += 1
                        break
                    elif idx[question, j] in sub[question, :3]:
                # We need to skip words already in the question.
                        continue
                    else:
                        # The correct label is not the precision@1
                        break
        print()
        print("Eval %4d/%d accuracy = %4.3f%%" % (correct, total,
                                                  correct * 100.0 / total))
    
 
    def analogy(self, w0, w1, w2):
        """Predict word w3 as in w0:w1 vs w2:w3."""
        wid = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
        idx = self._predict(wid)
        for c in [self._id2word[i] for i in idx[0, :]]:
            if c not in [w0, w1, w2]:
                print(c)
                return
        print("unknown")
  

    def nearby(self, words, num=20):
        """Prints out nearby words given a list of words."""
        near_words=[]
        ids = np.array([self._word2id.get(x, 0) for x in words])
        vals, idx = self._session.run(
            [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
        for i in xrange(len(words)):
    #       print("\n%s\n=====================================" % (words[i]))
            for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
                near_words.append((self._id2word[neighbor].decode("utf-8") , round(distance,3)))
    #         print("%-20s %6.4f" % (self._id2word[neighbor], distance))
        return near_words


    def get_simularities(self, main_word, words):
        """Prints out nearby words given a list of words."""
        main_id=[self._word2id.get(main_word, 0)]
        ids = np.array([self._word2id.get(x, 0) for x in words])
        sim,sim_1,sim_neg1= self._session.run([self._neighbors_sim, self._neighbors_sim_1,self._neighbors_sim_neg1], {self._nearby_word: main_id,self._neighbors: ids})

        return sim,sim_1,sim_neg1
    
    def nearby_custom2(self, word, num=20):
        """Prints out nearby words given a list of words. asdf"""
        near_words=[]
        ids = [self._word2id.get(word, 0)]
        vals, idx = self._session.run([self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
        neighbors=[self._id2word[x] for x in idx[0,:]]
        sims,sims_1,sims_neg_1=self.get_simularities(word,neighbors)
        for (neighbor, distance,sim,sim_1,sim_neg1) in zip(idx[0, :num], vals[0, :num],sims[0],sims_1[0],sims_neg_1[0]):
            near_words.append((self._id2word[neighbor].decode("utf-8") , round(sim,3),round(sim_1,3),round(sim_neg1,3)))
        columns = pd.MultiIndex.from_product([['near by cos sim'],['word', 'cos sim','cos sim x','cos sim y']])
        near_words=pd.DataFrame(near_words,columns=columns)

        near_1=[]
        ids = [self._word2id.get(word, 0)]
        vals, idx = self._session.run([self._nearby_val_1, self._nearby_idx_1], {self._nearby_word: ids})
        neighbors=[self._id2word[x] for x in idx[0,:]]
        sims,sims_1,sims_neg_1=self.get_simularities(word,neighbors)
        for (neighbor, distance,sim,sim_1,sim_neg1) in zip(idx[0, :num], vals[0, :num],sims[0],sims_1[0],sims_neg_1[0]):
            near_1.append((self._id2word[neighbor].decode("utf-8") , round(sim,3),round(sim_1,3),round(sim_neg1,3)))
        columns = pd.MultiIndex.from_product([['near by cos sim x'],['word', 'cos sim','cos sim x','cos sim y']])
        near_1=pd.DataFrame(near_1,columns=columns)

        near_neg1=[]
        ids = [self._word2id.get(word, 0)]
        vals, idx = self._session.run([self._nearby_val_neg1, self._nearby_idx_neg1], {self._nearby_word: ids})
        neighbors=[self._id2word[x] for x in idx[0,:]]
        sims,sims_1,sims_neg_1=self.get_simularities(word,neighbors)
        for (neighbor, distance,sim,sim_1,sim_neg1) in zip(idx[0, :num], vals[0, :num],sims[0],sims_1[0],sims_neg_1[0]):
            near_neg1.append((self._id2word[neighbor].decode("utf-8") , round(sim,3),round(sim_1,3),round(sim_neg1,3)))
        columns = pd.MultiIndex.from_product([['near by cos sim y'],['word', 'cos sim','cos sim x','cos sim y']])
        near_neg1=pd.DataFrame(near_neg1,columns=columns)


        far_neg1=[]
        ids = [self._word2id.get(word, 0)]
        vals, idx = self._session.run([self._far_val_neg1, self._far_idx_neg1], {self._nearby_word: ids})
        neighbors=[self._id2word[x] for x in idx[0,:]]
        sims,sims_1,sims_neg_1=self.get_simularities(word,neighbors)
        for (neighbor, distance,sim,sim_1,sim_neg1) in zip(idx[0, :num], vals[0, :num],sims[0],sims_1[0],sims_neg_1[0]):
            far_neg1.append((self._id2word[neighbor].decode("utf-8") ,round(sim,3),round(sim_1,3),round(sim_neg1,3)))
        columns = pd.MultiIndex.from_product([['far by cos sim y'],['word','cos sim','cos sim x','cos sim y']])
        far_neg1=pd.DataFrame(far_neg1,columns=columns)


        ids = [self._word2id.get(word, 0)]
        output=[]
        vals, idx = self._session.run([self._output_score, self._output_idx], {self._inp: ids})
        neighbors=[self._id2word[x] for x in idx[0,:]]
        sims,sims_1,sims_neg_1=self.get_simularities(word,neighbors)
        for (neighbor, distance,sim,sim_1,sim_neg1) in zip(idx[0, :num], vals[0, :num],sims[0],sims_1[0],sims_neg_1[0]):
            output.append((self._id2word[neighbor].decode("utf-8") , round(distance,3),round(sim,3),round(sim_1,3),round(sim_neg1,3)))
        columns = pd.MultiIndex.from_product([['output (w_i*c_j'],['word', 'score','cos sim','cos sim x','cos sim y']])
        output=pd.DataFrame(output,columns=columns)



        ids = [self._word2id.get(word, 0)]
        dot_product=[]
        vals, idx = self._session.run([self._dot_product_score, self._dot_product_idx], {self._inp: ids})
        neighbors=[self._id2word[x] for x in idx[0,:]]
        sims,sims_1,sims_neg_1=self.get_simularities(word,neighbors)
        for (neighbor, distance,sim,sim_1,sim_neg1) in zip(idx[0, :num], vals[0, :num],sims[0],sims_1[0],sims_neg_1[0]):
            dot_product.append((self._id2word[neighbor].decode("utf-8") , round(distance,3),round(sim,3),round(sim_1,3),round(sim_neg1,3)))
        columns = pd.MultiIndex.from_product([['w_i*w_j'],['word', 'score','cos sim','cos sim x','cos sim y']])
        dot_product=pd.DataFrame(dot_product,columns=columns)


        ids = [self._word2id.get(word, 0)]
        dot_product_x=[]
        vals, idx = self._session.run([self._dot_product_score_x, self._dot_product_idx_x], {self._inp: ids})
        neighbors=[self._id2word[x] for x in idx[0,:]]
        sims,sims_1,sims_neg_1=self.get_simularities(word,neighbors)
        for (neighbor, distance,sim,sim_1,sim_neg1) in zip(idx[0, :num], vals[0, :num],sims[0],sims_1[0],sims_neg_1[0]):
            dot_product_x.append((self._id2word[neighbor].decode("utf-8") , round(distance,3),round(sim,3),round(sim_1,3),round(sim_neg1,3)))
        columns = pd.MultiIndex.from_product([['x_i*x_j'],['word', 'score','cos sim','cos sim x','cos sim y']])
        dot_product_x=pd.DataFrame(dot_product_x,columns=columns)


        ids = [self._word2id.get(word, 0)]
        dot_product_y=[]
        vals, idx = self._session.run([self._dot_product_score_y, self._dot_product_idx_y], {self._inp: ids})
        neighbors=[self._id2word[x] for x in idx[0,:]]
        sims,sims_1,sims_neg_1=self.get_simularities(word,neighbors)
        for (neighbor, distance,sim,sim_1,sim_neg1) in zip(idx[0, :num], vals[0, :num],sims[0],sims_1[0],sims_neg_1[0]):
            dot_product_y.append((self._id2word[neighbor].decode("utf-8") , round(distance,3),round(sim,3),round(sim_1,3),round(sim_neg1,3)))
        columns = pd.MultiIndex.from_product([['y_i*y_j'],['word', 'score','cos sim','cos sim x','cos sim y']])
        dot_product_y=pd.DataFrame(dot_product_y,columns=columns)


        return pd.concat([near_words,near_1,near_neg1,far_neg1,output,dot_product_x,dot_product_y],axis=1)


    def _nearby_analogy(self,w0,w1,w2, num=20):
        self.analogy(w0,w1,w2)
        near_words=[]
        analogy = np.array([[self._word2id.get(w, 0) for w in [w0, w1, w2]]])
        vals, idx = self._session.run([self._nearby_analogy_val, self._nearby_analogy_idx], {self._analogy_a: analogy[:, 0],self._analogy_b: analogy[:, 1],self._analogy_c: analogy[:, 2]})
        for (neighbor, distance) in zip(idx[0, :num], vals[0, :num]):
            near_words.append((self._id2word[neighbor].decode("utf-8"), distance))

        far_words=[]
        vals, idx = self._session.run([self._far_analogy_val, self._far_analogy_idx], {self._analogy_a: analogy[:, 0],self._analogy_b: analogy[:, 1],self._analogy_c: analogy[:, 2]})
        vals*=-1
        for (neighbor, distance) in zip(idx[0, -num:], vals[0, -num:]):
            far_words.append((self._id2word[neighbor].decode("utf-8"), distance))


        ones=[]
        vals, idx = self._session.run([self._nearby_analogy_val_plus1, self._nearby_analogy_idx_plus1], {self._analogy_a: analogy[:, 0],self._analogy_b: analogy[:, 1],self._analogy_c: analogy[:, 2]})
        for (neighbor, distance) in zip(idx[0, :num], vals[0, :num]):
            ones.append((self._id2word[neighbor].decode("utf-8"), distance))

        far_ones=[]
        vals, idx = self._session.run([self._far_analogy_val_plus1, self._far_analogy_idx_plus1], {self._analogy_a: analogy[:, 0],self._analogy_b: analogy[:, 1],self._analogy_c: analogy[:, 2]})
        vals*=-1
        for (neighbor, distance) in zip(idx[0, -num:], vals[0, -num:]):
            far_ones.append((self._id2word[neighbor].decode("utf-8"), distance))


        neg_ones=[]
        vals, idx = self._session.run([self._nearby_analogy_val_neg1, self._nearby_analogy_idx_neg1], {self._analogy_a: analogy[:, 0],self._analogy_b: analogy[:, 1],self._analogy_c: analogy[:, 2]})
        for (neighbor, distance) in zip(idx[0, :num], vals[0, :num]):
            neg_ones.append((self._id2word[neighbor].decode("utf-8"), distance))

        far_neg_ones=[]
        vals, idx = self._session.run([self._far_analogy_val_neg1, self._far_analogy_idx_neg1], {self._analogy_a: analogy[:, 0],self._analogy_b: analogy[:, 1],self._analogy_c: analogy[:, 2]})
        vals*=-1
        for (neighbor, distance) in zip(idx[0, :num], vals[0, :num]):
            far_neg_ones.append((self._id2word[neighbor].decode("utf-8"), distance))


        table=pd.concat([pd.DataFrame(near_words,columns=['entire emb','cos sim']), pd.DataFrame(ones,columns=['+1s emb','cos sim']),pd.DataFrame(neg_ones,columns=['-1s emb','cos sim']),pd.DataFrame(far_words,columns=['entire far emb','cos sim']), pd.DataFrame(far_ones,columns=['+1s far emb','cos sim']),pd.DataFrame(far_neg_ones,columns=['-1s far emb','cos sim'])],axis=1)
        return table

    def get_embeddings(self,folder,precision=10,normal=False,txt=False,to_return=False):
        if normal:
            embs=self._session.run(self._nemb)
        else:
            embs=self._session.run(self._emb)
        if txt:
            np.savetxt(folder+"/"+"embeddings.txt", embs, fmt="%."+str(precision)+"f", delimiter=" ", newline="\n")
        if to_return:
            return embs
    
    def get_embeddings_1(self,folder,precision=10,normal=False,txt=False,to_return=False):
        if normal:
            embs=self._session.run(self._nemb_1)
        else:
            embs=self._session.run(self._emb[:,1::2])
        if txt:
            np.savetxt(folder+"/"+"embeddings_1.txt", embs, fmt="%."+str(precision)+"f", delimiter=" ", newline="\n")
        if to_return:
            return embs

    def get_embeddings_neg1(self,folder,precision=10,normal=False,txt=False,to_return=False):
        if normal:
            embs=self._session.run(self._nemb_neg1)
        else:
            embs=self._session.run(self._emb[:,::2])
        if txt:
            np.savetxt(folder+"/"+"embeddings_neg1.txt", embs, fmt="%."+str(precision)+"f", delimiter=" ", newline="\n")
        if to_return:
            return embs

    def plot_with_labels(self,low_dim_embs, labels,filename):
        assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(
              label,
              xy=(x, y),
              xytext=(5, 2),
              textcoords='offset points',
              ha='right',
              va='bottom')
        plt.show()
        plt.savefig(filename)
    
    def plot(self,mod='entire',plot_num=500):
    # pylint: disable=g-import-not-at-top

        embs=self._session.run(self._emb)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        if mod=='entire':
            low_dim_embs = tsne.fit_transform(embs[:plot_num, :])
        elif mod=='+1':
            low_dim_embs = tsne.fit_transform(embs[:plot_num, 1::2])
        elif mod=='cos sim y':
            low_dim_embs=tsne.fit_transform(embs[:plot_num, ::2])
        labels = [self._id2word[i] for i in xrange(plot_num)]
        self.plot_with_labels(low_dim_embs, labels, 'tsne.png')
  
    def plot_word(self,word,limit=50,mod='entire',plot_num=500):
        embs=self._session.run(self._emb)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        if mod=='entire':
            near_words=[]
            ids = [self._word2id.get(word, 0)]
            idx = self._session.run(self._nearby_idx, {self._nearby_word: ids})
            idx=idx[0][:limit]
            low_dim_embs = tsne.fit_transform(embs[idx, :])
        elif mod=='+1':
            near_1=[]
            ids = [self._word2id.get(word, 0)]
            idx = self._session.run(self._nearby_idx_1, {self._nearby_word: ids})
            idx=idx[0][:limit]
            low_dim_embs = tsne.fit_transform(embs[idx, 1::2])
        elif mod=='cos sim y':
            near_neg1=[]
            ids = [self._word2id.get(word, 0)]
            vals, idx = self._session.run([self._nearby_val_neg1, self._nearby_idx_neg1], {self._nearby_word: ids})
            idx=idx[0][:limit]
            low_dim_embs=tsne.fit_transform(embs[idx, ::2])
        elif mod=='-1 far':  
            far_neg1=[]
            ids = [self._word2id.get(word, 0)]
            vals, idx = self._session.run([self._far_val_neg1, self._far_idx_neg1], {self._nearby_word: ids})
            idx=idx[0][:limit]
            #append main_word since there is no main word in -1 fars
            idx=np.append(idx,[self._word2id.get(word, 0)])
            print(idx)
            low_dim_embs=tsne.fit_transform(embs[idx, ::2])
        labels = [self._id2word[i] for i in idx]
        self.plot_with_labels(low_dim_embs, labels, 'tsne.png')



class Tools():
    '''tools for SNGS+WT'''
    def __init__(self,model_path):
        self.model_path=model_path
        self.opts = Options()
        with tf.Graph().as_default(), tf.Session(config = config) as session:
            with tf.device("/cpu:0"):
                model = Word2Vec(self.opts, session)
                model.saver.restore(session, self.model_path)  

    def near_to_word(self,words):
        with tf.Graph().as_default(), tf.Session(config = config) as session:
            with tf.device("/cpu:0"):
                model = Word2Vec(self.opts, session)
                model.read_analogies() # Read analogy questions
                model.saver.restore(session, self.model_path) 
                for word in words:
                    print(word,':')
                    table=model.nearby_custom2(word.encode(),100000)
                    print(table)
                    table.to_excel(word+'.xlsx')
                 
    def generate_embs(self,mod='entire',folder='embeddings',precision=10,normal=False,txt=False,to_return=True):
        with tf.Graph().as_default(), tf.Session(config = config) as session:
            with tf.device("/cpu:0"):
                model = Word2Vec(self.opts, session)
                model.saver.restore(session, self.model_path)
                mods={'entire':model.get_embeddings,
                     '+1':model.get_embeddings_1,
                     'cos sim y':model.get_embeddings_neg1} 
                if mod=='all':
                    for k in mods.keys():
                        mods[k](folder,precision,normal,txt,to_return)
                else:
                    return mods[mod](folder,precision,normal,txt,to_return),model._word2id



if FLAGS.train=='True':
    """Train a word2vec model."""
    if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
        print("--train_data --eval_data and --save_path must be specified.")
        sys.exit(1)
    opts = Options()
    with tf.Graph().as_default(), tf.Session(config = config) as session:
        with tf.device("/cpu:0"):
            model = Word2Vec(opts, session)
            model.read_analogies() # Read analogy questions
            for _ in xrange(opts.epochs_to_train):
                print(session.run(model._words))
                model.train()  # Process one epoch
                model.eval()  # Eval analogies.
        # Perform a final save.
        model.saver.save(session,
                         os.path.join(opts.save_path, "model.ckpt"),
                         global_step=model.global_step)

if FLAGS.continue_train=='True':
    #continue to train
    print('hello')
    flags.FLAGS.__delattr__("epochs_to_train")
    flags.FLAGS.__delattr__("learning_rate")

    flags.DEFINE_integer(
        "epochs_to_train", 10,
        "Number of epochs to train. Each epoch processes the training data once "
        "completely.")
    flags.DEFINE_float("learning_rate", 0.052, "Initial learning rate.")
    FLAGS = flags.FLAGS

    opts = Options()
    with tf.Graph().as_default(), tf.Session(config = config) as session:
        with tf.device("/cpu:0"):
            model = Word2Vec(opts, session)
            model.read_analogies() # Read analogy questions
            model.saver.restore(session, savedmodel_paths[dataset])
            model.eval()  # Eval analogies.
            for _ in xrange(opts.epochs_to_train):
                model.train()  # Process one epoch
                model.eval()  # Eval analogies.
        # Perform a final save.
        model.saver.save(session,
                         os.path.join(opts.save_path, "model2.ckpt"),
                         global_step=model.global_step)
        if FLAGS.interactive:
          # E.g.,
          # [0]: model.analogy(b'france', b'paris', b'russia')
          # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
          _start_shell(locals())

tool=Tools(savedmodel_paths[dataset])
if FLAGS.gen_embs=='True':
    tool.generate_embs('all',embedding_folders[dataset],normal=False,txt=True,to_return=False)

if FLAGS.near_words_to is not None:
    words=FLAGS.near_words_to.lower()
    if ',' in words:
        words=words.replace(" ","").split(',')
    else:
        words=[words]
    tool.near_to_word(words)



if FLAGS.postag=='True':
    import nltk
    import numpy as np
    import pandas as pd
    from math import floor, ceil

    words=[]
    pos_tags=[]
    for sent in nltk.corpus.brown.tagged_sents(tagset='universal'):
        for word,pos_tag in sent:
            words.append(word.lower())
            pos_tags.append(pos_tag)

    embs,word2id=tool.generate_embs("entire",embedding_folders[dataset],normal=True,txt=False,to_return=True)

    words=np.array(words[:-1])
    pos_tags=pd.get_dummies(pd.Series(pos_tags[1:])).values

    X=np.array([word2id.get(x.encode()) for x in words])

    notna=~np.isnan(X.astype(float))
    X=X[notna].astype(int)
    pos_tags=pos_tags[notna]

    train_size = 0.8
    train_cnt = floor(X.shape[0] * train_size)
    X_train = X[0:train_cnt]
    y_train = pos_tags[0:train_cnt]
    X_test = X[train_cnt:]
    y_test = pos_tags[train_cnt:]

    for mode in ['cos sim y','+1','entire']:
        
            
        embs,word2id=tool.generate_embs(mode,embedding_folders[dataset],normal=True,txt=False,to_return=True)
        inputs = tf.placeholder(tf.float32, shape=(None, embs.shape[1]), name='inputs')
        label = tf.placeholder(tf.float32, shape=(None, pos_tags.shape[1]), name='labels')

        wo = tf.Variable(tf.random_normal([pos_tags.shape[1],embs.shape[1]], stddev=0.01), name='wo')
        bo = tf.Variable(tf.random_normal([pos_tags.shape[1], 1]), name='bo')
        yo = tf.transpose(tf.add(tf.matmul(wo, tf.transpose(inputs)), bo))

        lr = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yo, labels=label))
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        pred = tf.nn.softmax(yo)
        pred_label = tf.argmax(pred, 1)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.InteractiveSession(config=config)
        sess.run(init)

        batch_size=64
        learning_rate=0.8
        print(mode)
        for epoch in range(101):
            avg_cost = 0.0
            total_batch = int(len(X_train) / batch_size)
            x_batches = np.array_split(X_train, total_batch)
            y_batches = np.array_split(y_train, total_batch)
            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                batch_x=embs[batch_x]
                _, c = sess.run([optimizer, loss], feed_dict={lr:learning_rate, 
                                                              inputs: batch_x,
                                                              label: batch_y})
                avg_cost += c
            avg_cost /= X_train.shape[0]


            if epoch % 5 == 0:
                print("Epoch: {:3d}    Train Cost: {:.8f}".format(epoch, avg_cost))
            if epoch % 10 ==0:
                correct=0
                total=0
                for i in range(total_batch):
                    batch_x, batch_y = x_batches[i], y_batches[i]
                    batch_x=embs[batch_x]
                    corr_preds = correct_prediction.eval(feed_dict={inputs: batch_x, label: batch_y})
                    correct+=corr_preds.sum()
                    total+=corr_preds.shape[0]
                print("Train accuracy: {:3.2f}%".format((correct/total)*100.0),end=" ")


                correct=0
                total=0
                total_batch = int(len(X_test) / batch_size)
                x_batches = np.array_split(X_test, total_batch)
                y_batches = np.array_split(y_test, total_batch)
                for i in range(total_batch):
                    batch_x, batch_y = x_batches[i], y_batches[i]
                    batch_x=embs[batch_x]
                    corr_preds = correct_prediction.eval(feed_dict={inputs: batch_x, label: batch_y})
                    correct+=corr_preds.sum()
                    total+=corr_preds.shape[0]
                print("Test accuracy: {:3.2f}%".format((correct/total)*100.0))

