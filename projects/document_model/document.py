from typing import List
import re
import logging
import nltk
from nltk import pos_tag as nltk_pos_tagger
from .interval import Interval
from .common import get_shape_category_simple
from itertools import starmap, chain

LOGGER = logging.getLogger(__name__)

sentence_pattern = re.compile(
        r"""
            (\n\W*\([A-z]+\)) # newline then one or more letters in brackets
            |
            (\n\W*\([0-9]+\)) # newline then one or more numbers in brackets
            |
            (\n\W+\n)         # multiple blank lines
            |
            (\n([0-9]+\.)+)   # one or more numbers and dots after a newline
        """,
        flags=re.DOTALL and re.VERBOSE
    )


def setattrib(obj, name: str, value):
    setattr(obj, name, value)
    return obj


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
        2) How the tokeniser represents the token e.g. nltk.word_tokenize('"') == ['``']
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

    def __getitem__(self, item):
        return self.features[item]

    def __repr__(self):
        return 'Token({}, {}, {}, {}, {})'.format(self.text, self.start, self.end, self.features, self.label)


class Sentence(Interval):
    """ Interval corresponding to a Sentence"""

    def __init__(self, start: int, end: int):
        Interval.__init__(self, start, end)

    def __repr__(self):
        return 'Sentence({}, {})'.format(self.start, self.end)


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
        words, pos_tags = zip(*nltk.pos_tag(nltk.word_tokenize(text)))
        words, pos_tags = Document._retokenize(words, pos_tags)
        doc.tokens = Document._find_tokens(doc, words, pos_tags, text)
        doc.sentences = list(Document._find_sentences(text))

        doc.tokens = list(map(lambda item, idx: setattrib(item, 'index', idx), doc.tokens, range(len(doc.tokens))))
        doc.sentences = list(map(lambda item, idx: setattrib(item, 'index', idx), doc.sentences, range(len(doc.sentences))))
        return doc

    @classmethod
    def create_from_vectors(cls, words: List[str], sentences: List[Interval]=None, labels: List[str]=None):
        doc = Document()
        text = []
        offset = 0
        doc.sentences = []
        for sentence in sentences:
            text.append(' '.join(words[sentence.start:sentence.end + 1]) + ' ')
            doc.sentences.append(Interval(offset, offset + len(text[-1])))
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
    def _retokenize(word_tokens: List[str], post_tags:List[str]):
        """
        Correct NLTK tokenization. We separate symbols from words, such as quotes, -, *, etc
        :param word_tokens: list of strings(tokens) coming out of nltk.word_tokenize
        :return: new list of tokens
        """
        quotes = '’`"\'“”/\\'
        quote_list = list(quotes)
        strange_pos_tags = {'”': '\'\'', '“': '``', '-': ':'}
        known_symbols = re.escape('-*·')
        new_tokens, new_pos_tags = [], []
        for word_token, pos_tag in zip(word_tokens, post_tags):
            for new_token in [s for s in
                              re.split('([' + re.escape(quotes) + ']+)|(\n)|(^['+ known_symbols + '])|([' + known_symbols + ']$)',
                                       word_token) if s is not None]:
                if not len(new_token):
                    continue
                if any(q in new_token for q in quote_list):
                    new_pos_tags.append(new_token)
                elif new_token == '\n':
                    new_pos_tags.append('NL')
                else:
                    new_pos_tags.append(pos_tag)
                if new_pos_tags[-1] in strange_pos_tags:
                    new_pos_tags[-1] = strange_pos_tags[new_token]
                new_tokens.append(new_token)
        return new_tokens, new_pos_tags

    @staticmethod
    def _find_tokens(doc, word_tokens, pos_tags, text):
        """ Calculate the span of each token,
         find which element it belongs to and create a new Token instance """
        offset = 0
        tokens = []
        missing = []
        index = -1
        for token, pos_tag in zip(word_tokens, pos_tags):
            index += 1
            while offset < len(text) and (text[offset] == '\n' or text[offset] == ' '):
                if text[offset] == '\n':
                    tokens.append(Token(doc, offset, offset + 1, 'NL', get_shape_category('\n'), '\n'))
                offset += 1
            pos = text.find(token, offset, offset + max(50, len(token)))
            if pos > 0 and (pos - offset <= len(token) or len(missing) > 0):
                if missing:
                    start = tokens[-1].end if len(tokens) > 1 else 0
                    for m in missing:
                        while text[start] in [' ', '\n']:
                            if text[start] == '\n':
                                tokens.append(Token(doc, start, start + 1, 'NL', get_shape_category('\n'), '\n'))
                            start += 1
                        length = len(m[0]) if m[0] not in ['\'\'', '``'] else 1
                        tokens.append(Token(doc, start, start + length, m[1], get_shape_category(m[0]), m[0]))
                        start = start + length
                    missing = []
                tokens.append(Token(doc, pos, pos + len(token), pos_tag, get_shape_category(token), token))
                offset = pos + len(token)
            else:
                missing.append((token, pos_tag))
                LOGGER.debug('Token "{}" not found'.format(token))
        return tokens

    @staticmethod
    def _find_sentences(doc_text: str):
        """ yield Sentence objects each time a sentence is found in the text """
        splits = sentence_pattern.finditer(doc_text)
        indices = [0] + [split.start() for split in splits] + [len(doc_text)]
        slices = starmap(slice, zip(indices, indices[1:]))
        sents = chain.from_iterable(nltk.sent_tokenize(doc_text[slice_].replace('\n', ' ')) for slice_ in slices)
        text = doc_text.replace('\n', ' ')
        offset = 0
        for sent in sents:
            start = text.find(sent, offset)
            if start < 0:
                raise SentenceNotFoundError
            end = start + len(sent)
            offset = end
            yield Sentence(start, end)

    def get_sentence_tokens(self):
        all_tokens = []
        for sentence in self.sentences:
            all_tokens.append([token for token in self.tokens if sentence.overlaps(token)])
        return all_tokens
