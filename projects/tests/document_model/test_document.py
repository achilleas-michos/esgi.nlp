from unittest import TestCase
from projects.document_model import *


class TestParser(TestCase):
    def test_create(self):
        doc = Document.create_from_text('This is one sentence. And this is another!')
        # Test tokens
        self.assertEqual(len(doc.tokens), 10, 'Wrong tokens')
        self.assertEqual(doc.tokens[0], Interval(0, 4), 'Wrong indices')
        self.assertEqual(doc.tokens[-1], Interval(41, 42), 'Wrong indices')

        # Test Sentences
        self.assertEqual(len(doc.sentences), 2, 'Wrong sentences')
        self.assertEqual(doc.sentences[0], Interval(0, 21), 'Wrong indices')
        self.assertEqual(doc.sentences[-1], Interval(22, 42), 'Wrong indices')

        self.assertEqual(len(doc.sentences[0].tokens), 5, 'Wrong tokens in first sentence')
        self.assertEqual(len(doc.sentences[1].tokens), 5, 'Wrong tokens in second sentence')
