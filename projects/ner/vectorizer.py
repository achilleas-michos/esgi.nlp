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
        self.pos2index = {'PAD': 0, 'TO': 1, 'VBN': 2, "''": 3, 'WP': 4, 'UH': 5, 'VBG': 6, 'JJ': 7, 'VBZ': 8, '--': 9,
                          'VBP': 10, 'NN': 11, 'DT': 12, 'PRP': 13, ':': 14, 'WP$': 15, 'NNPS': 16, 'PRP$': 17,
                          'WDT': 18, '(': 19, ')': 20, '.': 21, ',': 22, '``': 23, '$': 24, 'RB': 25, 'RBR': 26,
                          'RBS': 27, 'VBD': 28, 'IN': 29, 'FW': 30, 'RP': 31, 'JJR': 32, 'JJS': 33, 'PDT': 34, 'MD': 35,
                          'VB': 36, 'WRB': 37, 'NNP': 38, 'EX': 39, 'NNS': 40, 'SYM': 41, 'CC': 42, 'CD': 43, 'POS': 44,
                          'LS': 45}
        self.shape2index = {'PAD': 0, 'ALL-LOWER': 1, 'ALL-UPPER': 2, 'FIRST-UPPER': 3, 'MISC': 4}
        self.label2index = {'O': 0, 'PER': 1, 'I-PER': 1, 'B-PER': 1, 'LOC': 2, 'I-LOC': 2, 'B-LOC': 2,
                            'ORG': 3, 'I-ORG': 3, 'B-ORG': 3, 'MISC': 4, 'I-MISC': 4, 'B-MISC': 4}
        self.labels = ['O', 'PER', 'LOC', 'ORG', 'MISC']
        self.word_embeddings = KeyedVectors.load_word2vec_format(word_embedding_path)

    def encode_features(self, documents: List[Document], max_length=100):
        """
        Creates a feature matrix for all documents in the sample list
        :param documents: list of all samples as document objects
        :return: list of numpy arrays (one item per document)
        """
        no_sentences = sum([len(document.sentences) for document in documents])
        words, pos, shapes = np.zeros((no_sentences, max_length)), np.zeros((no_sentences, max_length)),\
                             np.zeros((no_sentences, max_length))
        i = 0
        for document in tqdm(documents):
            for sentence in document.sentences:
                for j, token in enumerate(sentence.tokens):
                    if j == max_length:
                        break
                    if token.text.lower() in self.word_embeddings.index2word:
                        words[i][j] = self.word_embeddings.index2word.index(token.text.lower())
                    else:
                        words[i][j] = 0
                    pos[i][j] = self.pos2index[token.features['pos']]
                    shapes[i][j] = self.shape2index[token.features['shape']]
                i += 1

        return words, pos, shapes

    def encode_annotations(self, documents: List[Document]):
        """
        Creates the Y matrix representing the annotations (or true positives) of a list of documents
        :param documents: list of documents to be converted in annotations vector
        :return: numpy array (one item per document)
        """
        annotations = []
        for i, document in tqdm(enumerate(documents)):
            for sentence in document.sentences:
                sentence_annotations = np.zeros((len(sentence), 1))
                for j, token in enumerate(sentence.tokens):
                    sentence_annotations[j] = self.label2index[token.label]
                annotations.append(sentence_annotations)
        return annotations
