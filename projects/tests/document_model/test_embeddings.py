import os
from unittest import TestCase
from data import DATA_DIR
from projects.document_model import *


class TestEmbeddigns(TestCase):
    def test_load_embeddings(self):
        filename = os.path.join(DATA_DIR, 'embeddings', 'glove.6B.50d.w2v.txt')
        embeddings = load_embeddings(filename, source='gensim', binary=False)

        self.assertIsNotNone(embeddings, 'Embeddings not loaded')
