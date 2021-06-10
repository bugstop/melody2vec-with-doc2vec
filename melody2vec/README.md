## melody2vec with doc2vec

1. **run Tatsunori Hirai's [melody2vec](https://github.com/TatsunoriHirai/Melody2vec) for preprocessing**  
   files in folder `/midi` and `/melody-segmented` are needed in next steps
2. **run `melody.py` to prepare the dataset**  
   `melody.json` includes melody indexes, segmented melodies, and midi filenames
3. **run `doc2vec.py` to train the doc2vec model**  
   paragraph vectors of training melodies are shown in `doc2vec.json`
4. **some downstream tasks are shown [here](../downstream%20tasks)**  
   just demos XD
