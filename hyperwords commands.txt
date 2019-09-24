cut -d"'" -f2 tmp/vocab.txt > embeddings/words.txt
paste words.txt embeddings/embeddings.txt -d" " > embeddings/emb.words
paste words.txt embeddings/embeddings_1.txt -d" " > embeddings/emb_1.words
paste words.txt embeddings/embeddings_neg1.txt -d" " > embeddings/emb_neg1.words


python hyperwords/hyperwords/text2numpy.py embeddings/emb.words
python hyperwords/hyperwords/analogy_eval.py SGNS embeddings/emb hyperwords/testsets/analogy/google.txt
python hyperwords/hyperwords/analogy_eval.py SGNS embeddings/emb hyperwords/testsets/analogy/msr.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings/emb hyperwords/testsets/ws/ws353.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings/emb hyperwords/testsets/ws/bruni_men.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings/emb hyperwords/testsets/ws/radinsky_mturk.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings/emb hyperwords/testsets/ws/luong_rare.txt


#1
python hyperwords/hyperwords/text2numpy.py embeddings/emb_1.words
python hyperwords/hyperwords/analogy_eval.py SGNS embeddings/emb_1 hyperwords/testsets/analogy/google.txt
python hyperwords/hyperwords/analogy_eval.py SGNS embeddings/emb_1 hyperwords/testsets/analogy/msr.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings/emb_1 hyperwords/testsets/ws/ws353.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings/emb_1 hyperwords/testsets/ws/bruni_men.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings/emb_1 hyperwords/testsets/ws/radinsky_mturk.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings/emb_1 hyperwords/testsets/ws/luong_rare.txt

#neg1
python hyperwords/hyperwords/text2numpy.py embeddings/emb_neg1.words
python hyperwords/hyperwords/analogy_eval.py SGNS embeddings/emb_neg1 hyperwords/testsets/analogy/google.txt
python hyperwords/hyperwords/analogy_eval.py SGNS embeddings/emb_neg1 hyperwords/testsets/analogy/msr.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings/emb_neg1 hyperwords/testsets/ws/ws353.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings/emb_neg1 hyperwords/testsets/ws/bruni_men.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings/emb_neg1 hyperwords/testsets/ws/radinsky_mturk.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings/emb_neg1 hyperwords/testsets/ws/luong_rare.txt





cut -d"'" -f2 tmp_fil9/vocab.txt > embeddings_fil9/words.txt
paste embeddings_fil9/words.txt embeddings_fil9/embeddings.txt -d" " > embeddings_fil9/emb.words
python hyperwords/hyperwords/text2numpy.py embeddings_fil9/emb.words
python hyperwords/hyperwords/analogy_eval.py SGNS embeddings_fil9/emb hyperwords/testsets/analogy/google.txt
python hyperwords/hyperwords/analogy_eval.py SGNS embeddings_fil9/emb hyperwords/testsets/analogy/msr.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings_fil9/emb hyperwords/testsets/ws/ws353.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings_fil9/emb hyperwords/testsets/ws/bruni_men.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings_fil9/emb hyperwords/testsets/ws/radinsky_mturk.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings_fil9/emb hyperwords/testsets/ws/luong_rare.txt

rm embeddings_fil9/emb.words
rm embeddings_fil9/emb.words.npy
rm embeddings_fil9/emb.words.vocab


paste embeddings_fil9/words.txt embeddings_fil9/embeddings_1.txt -d" " > embeddings_fil9/emb_1.words
python hyperwords/hyperwords/text2numpy.py embeddings_fil9/emb_1.words
python hyperwords/hyperwords/analogy_eval.py SGNS embeddings_fil9/emb_1 hyperwords/testsets/analogy/google.txt
python hyperwords/hyperwords/analogy_eval.py SGNS embeddings_fil9/emb_1 hyperwords/testsets/analogy/msr.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings_fil9/emb_1 hyperwords/testsets/ws/ws353.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings_fil9/emb_1 hyperwords/testsets/ws/bruni_men.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings_fil9/emb_1 hyperwords/testsets/ws/radinsky_mturk.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings_fil9/emb_1 hyperwords/testsets/ws/luong_rare.txt

rm embeddings_fil9/emb_1.words
rm embeddings_fil9/emb_1.words.npy
rm embeddings_fil9/emb_1.words.vocab




paste embeddings_fil9/words.txt embeddings_fil9/embeddings_neg1.txt -d" " > embeddings_fil9/emb_neg1.words
python hyperwords/hyperwords/text2numpy.py embeddings_fil9/emb_neg1.words
python hyperwords/hyperwords/analogy_eval.py SGNS embeddings_fil9/emb_neg1 hyperwords/testsets/analogy/google.txt
python hyperwords/hyperwords/analogy_eval.py SGNS embeddings_fil9/emb_neg1 hyperwords/testsets/analogy/msr.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings_fil9/emb_neg1 hyperwords/testsets/ws/ws353.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings_fil9/emb_neg1 hyperwords/testsets/ws/bruni_men.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings_fil9/emb_neg1 hyperwords/testsets/ws/radinsky_mturk.txt
python hyperwords/hyperwords/ws_eval.py SGNS embeddings_fil9/emb_neg1 hyperwords/testsets/ws/luong_rare.txt

rm embeddings_fil9/emb_neg1.words
rm embeddings_fil9/emb_neg1.words.npy
rm embeddings_fil9/emb_neg1.words.vocab