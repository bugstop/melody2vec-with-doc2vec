import gensim
import json
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

with open('melody.json') as f:
    melody = json.load(f)

document = gensim.models.doc2vec.TaggedDocument
data = [document(m[1].split(' '), tags=[m[0]]) for m in melody]
print(len(data), data[0])

# doc2vec parameters
vector_size = 256
window_size = 8
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 20000
dm = 0             # 0 = dbow; 1 = dmpv
worker_count = 8   # number of parallel processes
pretrained = 'w2v.model'

model = gensim.models.Doc2Vec(
    data, vector_size=vector_size, window=window_size,
    min_count=min_count, sample=sampling_threshold,
    workers=worker_count, negative=negative_size,
    hs=0, dm=dm, dbow_words=1, dm_concat=1,
    pretrained_emb=pretrained, epochs=train_epoch
)

model.save('d2v.bin')

# 2021-02-25 17:51:03,236 : INFO : EPOCH - 300 :
#     training on 957629 raw words (852631 effective
#     words) took 15.7s, 54480 effective words/s
# 2021-02-25 17:51:03,236 : INFO : training on a
#     287288700 raw words (255790174 effective words)
#     took 4829.6s, 52964 effective words/s
