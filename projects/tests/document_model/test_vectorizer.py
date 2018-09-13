import os
from unittest import TestCase
from data import DATA_DIR
from projects.document_model import *
from projects.ner import Vectorizer


class TestVectorizer(TestCase):
    def test_EnglishNerParser(self):
        docs = EnglishNerParser().read_file(os.path.join(DATA_DIR, 'ner', 'eng.test.txt'))
        vectorizer = Vectorizer(word_embedding_path=os.path.join(DATA_DIR, 'embeddings', 'glove.6B.50d.w2v.txt'))

        feature_vectors = vectorizer.encode_features(docs)
        label_vectors = vectorizer.encode_annotations(docs)
        self.assertEqual(len(feature_vectors), len(label_vectors), 'Error')
