from typing import List
import re
import logging
import nltk
from nltk import pos_tag as nltk_pos_tagger
from .interval import Interval
from .common import get_shape_category_simple


LOGGER = logging.getLogger(__name__)


class TokenNotFoundException(Exception):
    """Raised when DocBookParser cannot find one or more tokens """
    pass


class SentenceNotFoundError(Exception):
    pass


class Token(Interval):
    """ A Interval representing word like units of text with a dictionary of features """
    def __init__(self, document, start: int, end: int, pos: str, shape: int, text: str, label: str=None):
        """
        Note that a token has 2 text representations.
        1) How the text appears in the original document e.g. doc.text[token.start:token.end]
        2) How the tokenizer represents the token e.g. nltk.word_tokenize('"') == ['``']
        :param document: the document object containing the token
        :param start: start of token in document text
        :param end: end of token in document text
        :param pos: part of speach of the token
        :param shape: integer label describing the shape of the token (particular to cog+)
        :param text: this is the text representation of token
        """
        Interval.__init__(self, start, end)
        self._doc = document
        self.features = {'pos': pos, 'shape': shape}
        self.token_text = text
        self.label = label
        self.source = None

    @property
    def text(self):
        return self._doc.text[self.start:self.end]

    @property
    def pos(self):
        return self.features['pos']

    @property
    def shape(self):
        return self.features['shape']

    def __repr__(self):
        return 'Token({}, {}, {}, {}, {})'.format(self.text, self.start, self.end, self.features, self.label)


class Sentence(Interval):
    """ Interval corresponding to a Sentence"""

    def __init__(self, doc, start: int, end: int):
        Interval.__init__(self, start, end)
        self._doc = doc

    def __repr__(self):
        return 'Sentence({}, {})'.format(self.start, self.end)

    @property
    def tokens(self):
        return [token for token in self._doc.tokens if self.overlaps(token)]


class Document:
    """
    A document is a combination of text and the positions of the tags and elements in that text.
    """
    def __init__(self):
        self.text = None
        self.tokens = None
        self.sentences = None

    @classmethod
    def create_from_text(cls,  text: str=None):
        """
        :param text: document text as a string
        :param tags: list of Tag objects
        """
        doc = Document()
        if text is None or text == '':
            return doc
        doc.text = text
        doc.tokens = Document._find_tokens(doc, text)
        doc.sentences = Document._find_sentences(doc, text)
        return doc

    @classmethod
    def create_from_vectors(cls, words: List[str], sentences: List[Interval]=None, labels: List[str]=None):
        doc = Document()
        offset = 0
        text, doc.sentences = [], []
        for sentence in sentences:
            text.append(' '.join(words[sentence.start:sentence.end + 1]) + ' ')
            doc.sentences.append(Sentence(doc, offset, offset + len(text[-1])))
            offset += len(text[-1])
        doc.text = ''.join(text)

        offset = 0
        doc.tokens = []
        for word_pos, label in zip(nltk_pos_tagger(words), labels):
            word = word_pos[0]
            pos_tag = word_pos[1]
            pos = doc.text.find(word, offset)
            if pos >= 0:
                offset = pos + len(word)
                doc.tokens.append(Token(doc, pos, offset, pos_tag, get_shape_category_simple(word), word, label=label))
        return doc

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            return self.tokens[key].text

    def __repr__(self):
        return 'Document(tokens={}, sentences={} text={})'.format(self.tokens, self.sentences, self.text)

    @staticmethod
    def _find_tokens(doc, text):
        """ Calculate the span of each token,
         find which element it belongs to and create a new Token instance """
        word_tokens, pos_tags = zip(*nltk.pos_tag(nltk.word_tokenize(text)))

        offset = 0
        tokens, missing = [], []
        for token, pos_tag in zip(word_tokens, pos_tags):
            while offset < len(text) and (text[offset] == '\n' or text[offset] == ' '):
                if text[offset] == '\n':
                    tokens.append(Token(doc, offset, offset + 1, 'NL', get_shape_category_simple('\n'), '\n'))
                offset += 1
            pos = text.find(token, offset, offset + max(50, len(token)))
            if pos > -1:
                if missing:
                    start = tokens[-1].end if len(tokens) > 1 else 0
                    for m in missing:
                        while text[start] in [' ', '\n']:
                            if text[start] == '\n':
                                tokens.append(Token(doc, start, start + 1, 'NL', get_shape_category_simple('\n'), '\n'))
                            start += 1
                        length = len(m[0]) if m[0] not in ['\'\'', '``'] else 1
                        tokens.append(Token(doc, start, start + length, m[1], get_shape_category_simple(m[0]), m[0]))
                        start = start + length
                    missing = []
                tokens.append(Token(doc, pos, pos + len(token), pos_tag, get_shape_category_simple(token), token))
                offset = pos + len(token)
            else:
                missing.append((token, pos_tag))
                LOGGER.debug('Token "{}" not found'.format(token))
        return tokens

    @staticmethod
    def _find_sentences(doc, doc_text: str):
        sentence_objects = []
        offset = 0
        for sentence in nltk.sent_tokenize(doc_text):
            pos = doc_text.find(sentence, offset)
            if pos > -1:
                sentence_objects.append(Sentence(doc, pos, pos + len(sentence)))
        return sentence_objects

