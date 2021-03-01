import gensim
import json

with open('melody.json') as f:
    melody = json.load(f)

model = gensim.models.Doc2Vec.load('d2v.bin')

# inference hyper-parameters
alpha = 0.01
epoch = 1000

sims = model.docvecs.most_similar([model.docvecs[0]], topn=10)
print(sims)

inferred = model.infer_vector(melody[0][1].split(' '), alpha=alpha, steps=epoch)
sims = model.docvecs.most_similar([inferred], topn=10)
print(sims)

print(melody[0][-1], melody[245][-1], melody[9533][-1])

# [(0, 1.0), (8811, 0.9978615045547485), (8199, 0.9833886027336121), (245, 0.9817537665367126),
#     (9533, 0.7279029488563538), (6413, 0.6909092664718628), (2683, 0.6771758794784546),
#     (1442, 0.6739320158958435), (5882, 0.6705861687660217), (4714, 0.6636027097702026)]
# [(8811, 0.9855718612670898), (0, 0.983768880367279), (8199, 0.9715628623962402), (245, 0.970552384853363),
#     (9533, 0.7039425373077393), (6413, 0.645868182182312), (5884, 0.6291266083717346),
#     (5882, 0.6259562969207764), (4714, 0.6193612217903137), (4073, 0.6068050861358643)]
# 000e12872af8de95ab2f3b4957da5805.mid 055df847de7ea214a5f57b177e362d22.mid e31b00bc057683065e307eef3fe07928.mid
