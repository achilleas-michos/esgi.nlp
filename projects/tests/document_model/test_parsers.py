import os
from unittest import TestCase
from data import DATA_DIR
from projects.document_model import *


class TestParser(TestCase):
    def test_EnglishNerParser(self):
        doc, labels = EnglishNerParser().read_file(os.path.join(DATA_DIR, 'ner', 'eng.test.txt'))
        self.assertEqual(len(doc), 216, 'Some documents were not extracted')