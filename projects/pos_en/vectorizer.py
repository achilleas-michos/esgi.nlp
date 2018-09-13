import numpy as np
import logging
from gensim.models import KeyedVectors
from typing import List
from tqdm import tqdm
from projects.document_model import Document

LOGGER = logging.getLogger(__name__)


class Vectorizer:
    """ Transform a string into a vector representation"""
    def __init__(self, word_embedding_path: str):
        """
        :param word_embedding_path: path to gensim embedding file
        """
        self.label2index = {'PAD': 0, 'TO': 1, 'VBN': 2, "''": 3, 'WP': 4, 'UH': 5, 'VBG': 6, 'JJ': 7, 'VBZ': 8, '--': 9,
                            'VBP': 10, 'NN': 11, 'DT': 12, 'PRP': 13, ':': 14, 'WP$': 15, 'NNPS': 16, 'PRP$': 17,
                            'WDT': 18, '(': 19, ')': 20, '.': 21, ',': 22, '``': 23, '$': 24, 'RB': 25, 'RBR': 26,
                            'RBS': 27, 'VBD': 28, 'IN': 29, 'FW': 30, 'RP': 31, 'JJR': 32, 'JJS': 33, 'PDT': 34, 'MD': 35,
                            'VB': 36, 'WRB': 37, 'NNP': 38, 'EX': 39, 'NNS': 40, 'SYM': 41, 'CC': 42, 'CD': 43, 'POS': 44,
                            'LS': 45, '"': 46}
        self.shape2index = {'PAD': 0, 'ALL-LOWER': 1, 'ALL-UPPER': 2, 'FIRST-UPPER': 3, 'MISC': 4}
        self.labels = self.label2index.keys()
        self.word_embeddings = KeyedVectors.load_word2vec_format(word_embedding_path)

    def encode_features(self, documents: List[Document],):
        """
        Creates a feature matrix for all documents in the sample list
        :param documents: list of all samples as document objects
        :return: list of numpy arrays (one item per document)
        """
        features = {'words': [], 'shape': []}
        for i, document in tqdm(enumerate(documents)):
            for sentence in document.get_sentence_tokens():
                features['words'].append(np.zeros(len(sentence)))
                features['shape'].append(np.zeros(len(sentence)))

                for j, token in enumerate(sentence):
                    if token.text.lower() in self.word_embeddings.index2word:
                        features['words'][-1][j] = self.word_embeddings.index2word.index(token.text.lower())
                    else:
                        features['words'][-1][j] = 0
                    features['shape'][-1][j] = self.shape2index[token.features['shape']]

        return features

    def encode_annotations(self, documents: List[Document]):
        """
        Creates the Y matrix representing the annotations (or true positives) of a list of documents
        :param documents: list of documents to be converted in annotations vector
        :return: numpy array (one item per document)
        """
        annotations = []
        for i, document in tqdm(enumerate(documents)):
            for sentence in document.get_sentence_tokens():
                sentence_annotations = np.zeros((len(sentence), 1))
                for j, token in enumerate(sentence):
                    sentence_annotations[j] = self.label2index[token.label]
                annotations.append(sentence_annotations)
        return annotations
