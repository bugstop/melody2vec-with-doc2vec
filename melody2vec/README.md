## melody2vec with doc2vec

1. run [Tatsunori Hirai's](https://github.com/TatsunoriHirai/Melody2vec) melody2vec
2. run `melody.py` to get everything stored in `melody.json` (index, segmented melody, midi filename)
3. run `doc2vec.py` to train the doc2vec model and get the paragraph vector of songs
4. run `inferene.py` to compute new paragraph vectors and similarities
5. run `fetch.py` to get original midi file given its index in `melody.json`

