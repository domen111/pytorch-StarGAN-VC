import numpy as np
import os
import glob

class SpeakerEmbedding:
    def __init__(self, embedding_dir='data/speakers_embedding'):
        self.embedding_dir = embedding_dir
        # self.speakers = np.array([['p1', 'p2']])
    
    def transform(self, speaker):
        # return self.speakers == speaker
        files = glob.glob(os.path.join(self.embedding_dir, speaker + '/*.npy'))
        embeddings = [np.load(f) for f in files]
        return np.mean(embeddings, axis=0)
