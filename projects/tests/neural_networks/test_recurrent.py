import os
from unittest import TestCase
from data import DATA_DIR
from projects.neural_networks import Recurrent
from projects.ner import Vectorizer


class TestRecurrent(TestCase):
    def test_build(self):
        vectorizer = Vectorizer(word_embedding_path=os.path.join(DATA_DIR, 'embeddings', 'glove.6B.50d.w2v.txt'))

        model = Recurrent.build('bilstm_lstm', word_embeddings=vectorizer.word_embeddings,
                                input_shape={'pos': (len(vectorizer.pos2index), 10),
                                             'shape': (len(vectorizer.shape2index), 2)},
                                out_shape=len(vectorizer.labels),
                                units=100, dropout_rate=0.5)

    def test_functiona(self):
        from keras.preprocessing import sequence
        from keras.models import Sequential
        from keras.layers import Dense, Embedding, Flatten, Dropout
        from keras.datasets import imdb
        from sklearn import metrics

        vocab_size = 20000
        maxlen = 100  # cut texts after this number of words (among top max_features most common words)
        batch_size = 32

        print('Loading data...')
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
        print('Train sequences: {}'.format(len(x_train)))
        print('Train sequences: {}'.format(len(x_test)))

        print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

        print('x_train shape: {}'.format(x_train.shape))
        print('y_train shape: {}'.format(y_train.shape))
        print('x_test shape: {}'.format(x_test.shape))
        print('y_test shape: {}'.format(y_test.shape))

        print('Build MLP model...')
        model = Sequential()
        model.add(Embedding(vocab_size, 50, input_length=maxlen))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(units=100, activation='relu', name='dense_layer_in'))
        model.add(Dense(units=1, activation='sigmoid', name='dense_layer_out'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('Train...')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=1,
                  validation_data=(x_test, y_test))
        print('Test score:', score)
        print('Test accuracy:', acc)

        y_pred = model.predict_classes(x_test, batch_size=1, verbose=0)
        print(metrics.classification_report(y_test, y_pred.flatten()))