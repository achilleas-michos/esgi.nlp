import logging
import time
import gensim
import pickle

LOGGER = logging.getLogger(__name__)


class Embedding:
    """
    Some embeddings are gensim objects and some are python dict objects so this is an adapter to
     give them the same interface.
    """
    # TODO: Maybe all the embedding files should be in the same format. Then we can get rid of this
    def __init__(self, obj, size):
        self.obj = obj
        self.size = size

    def __getitem__(self, item):
        return self.obj[item]

    def __contains__(self, item):
        return item in self.obj

    def __len__(self):
        return self.size


def load_embeddings(filename: str, source: str = 'gensim', use_norm: bool = False) -> object:
    """
    :rtype: list
    """
    try:
        time1 = time.time()
        if source == 'gensim':
            embeddings = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
            if use_norm:
                embeddings.init_sims(replace=True)
            vocab_size = len(embeddings.index2word)
            emb_size = embeddings.vector_size
        elif source == 'pickle':
            with open(filename, 'rb') as fp:
                embeddings = pickle.load(fp)
            vocab_size = len(embeddings)
            emb_size = len(embeddings[list(embeddings.keys())[0]])
    except:
        raise Exception('Embeddings file could not be loaded')

    return Embedding(embeddings, emb_size)
