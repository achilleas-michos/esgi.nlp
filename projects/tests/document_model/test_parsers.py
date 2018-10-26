import os
from unittest import TestCase
from data import DATA_DIR
from projects.document_model import *


class TestParser(TestCase):
    def test_EnglishNerParser(self):
        doc = EnglishNerParser().read_file(os.path.join(DATA_DIR, 'ner', 'eng.test.txt'))
        self.assertEqual(len(doc), 216, 'Some documents were not extracted')

    def test_EnglishPosParser(self):
        doc = EnglishPosParser().read_file(os.path.join(DATA_DIR, 'ner', 'eng.test.txt'))
        self.assertEqual(len(doc), 216, 'Some documents were not extracted')

    def test_AmazonParser(self):
        doc = AmazonReviewParser().read_file(os.path.join(DATA_DIR, 'reviews', 'Automotive_5_train.json'))
        self.assertEqual(len(doc), 15000, 'Some documents were not extracted')
