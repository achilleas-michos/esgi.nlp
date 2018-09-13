import os
from data import DATA_DIR
from sklearn.metrics import classification_report
from projects.neural_networks import NeuralNetwork, Recurrent
from projects.document_model import EnglishNerParser
from projects.ner import Vectorizer


if __name__ == '__main__':
    print('Loading models')
    vectorizer = Vectorizer(word_embedding_path=os.path.join(DATA_DIR, 'embeddings', 'glove.6B.50d.w2v.txt'))

    if False:
        model = Recurrent.build('bilstm_lstm', word_embeddings=vectorizer.word_embeddings,
                                input_shape={'pos': (len(vectorizer.pos2index), 10),
                                             'shape': (len(vectorizer.shape2index), 2)},
                                out_shape=len(vectorizer.labels),
                                units=100, dropout_rate=0.5)

        model.load_weights(os.path.join(DATA_DIR, 'models', 'ner_weights.h5'))
    else:
        model = Recurrent.load(os.path.join(DATA_DIR, 'models', 'ner.h5'))

    print('Reading test data')
    documents = EnglishNerParser().read_file(os.path.join(DATA_DIR, 'ner', 'eng.test.txt'))
    features = vectorizer.encode_features(documents)
    labels = vectorizer.encode_annotations(documents)

    print('Loaded {} data samples'.format(len(labels)))
    print('================= Classificatin report ========================')

    y_pred, y_dev = [], []
    for i in range(len(labels)):
        y_p = model.predict([features['words'][i], features['pos'][i], features['shape'][i]],
                                    batch_size=1, verbose=0)
        y_p = NeuralNetwork.probas_to_classes(y_p)
        y_pred.extend(y_p.flatten())
        y_dev.extend(labels[i].flatten())

    print(classification_report(y_dev, y_pred))

