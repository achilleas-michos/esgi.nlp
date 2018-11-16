from typing import  List
from .document import Document, Interval
from nltk.tokenize import word_tokenize
import json
from tqdm import tqdm


class Parser(object):
    def create(self):
        return self

    def read_file(self, filename: str) -> List[Document]:
        with open(filename, 'r', encoding='utf-8') as fp:
            content = fp.read()
        return self.read(content)

    def read(self, content: str):
        raise NotImplemented


class SimpleTextParser(Parser):
    def read(self, content: str) -> Document:
        return Document(content)


class EnglishNerParser(Parser):
    def read(self, content: str):
        documents = []
        for text in tqdm(content.split('-DOCSTART- -X- O O')):  # Split documents
            if text == '':
                continue
            words, sentences, labels = [], [], []
            sent_start, idx = 0, 0
            for line in text.splitlines():  # Each line has one word
                if line.split():
                    words.append(line.split()[0])
                    labels.append(line.split()[3])
                    idx += 1
                elif sent_start < idx:  # Empty line means new sentence
                    sentences.append(Interval(sent_start, idx - 1))
                    sent_start = idx
            documents.append(Document().create_from_vectors(words, sentences, labels))
        return documents


class EnglishPosParser(Parser):
    def read(self, content: str) -> Document:
        documents = []
        for text in tqdm(content.split('-DOCSTART- -X- O O')):  # Split documents
            if text == '':
                continue
            words, sentences, labels = [], [], []
            sent_start, idx = 0, 0
            for line in text.splitlines():  # Each line has one word
                if line.split():
                    words.append(line.split()[0])
                    labels.append(line.split()[1])
                    idx += 1
                elif sent_start < idx:  # Empty line means new sentence
                    sentences.append(Interval(sent_start, idx - 1))
                    sent_start = idx
            documents.append(Document().create_from_vectors(words, sentences, labels))
        return documents


class AmazonReviewParser(Parser):
    def read(self, content: str) -> Document:
        documents = []
        for line in tqdm(content.splitlines()):
            sentences = []
            data = json.loads(line)
            text = data['reviewText'] + ' ' + data['summary']
            words = word_tokenize(text)
            labels = [data['overall']] * len(text)
            sentences.append(Interval(0, len(text)-1))
            documents.append(Document().create_from_vectors(words, sentences, labels))
        return documents
