import gensim
import json

with open('melody.json') as f:
    melody = json.load(f)

model = gensim.models.Doc2Vec.load('d2v.bin')

# inference hyper-parameters
alpha = 0.01
epoch = 1000

while True:
    x = input('-> ')
    if not x:
        break
    x = int(x)

    sims = model.docvecs.most_similar(positive=[model.docvecs[x]], topn=10)
    print(sims)
    sims = model.docvecs.most_similar(negative=[model.docvecs[x]], topn=10)
    print(sims)

    inferred = model.infer_vector(melody[x][1].split(' '), alpha=alpha, steps=epoch)
    # print(type(inferred), inferred.tolist())

    sims = model.docvecs.most_similar(positive=[inferred], topn=10)
    print(sims)
    sims = model.docvecs.most_similar(negative=[inferred], topn=10)
    print(sims)
