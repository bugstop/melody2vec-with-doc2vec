## melody2vec with doc2vec

1. run Tatsunori Hirai's [melody2vec](https://github.com/TatsunoriHirai/Melody2vec) (files in folder `/midi` and `/melody-segmented` are needed in next steps)
2. run `melody.py` to get everything stored in `melody.json` (index, segmented melody, midi filename)
3. run `doc2vec.py` to train the doc2vec model and get the paragraph vectors of songs
4. run `inference.py` to infer new paragraph vectors and compute similarities
5. run `fetch.py` to get original midi file given its index in `melody.json`
