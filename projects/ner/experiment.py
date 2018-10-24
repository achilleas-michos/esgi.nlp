import os
from data import DATA_DIR
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from projects.neural_networks import NeuralNetwork, Recurrent
from projects.document_model import EnglishNerParser
from projects.ner import Vectorizer


def embeddings_test(self):
    print('TOP-10 SIMILAR WORDS BASED ON A SINGLE WORD')
    for word in ['dog','france','computer','king','batman']:
        print('{0} --> {1}'.format(word, ', '.join([word_sim for (word_sim, cos_sim) in self.word_embeddings.most_similar(positive=[word])])))

    print('TOP-10 SIMILAR WORDS BASED ON WORD PAIR')
    for pair in [['may','could'], ['may','june'], ['duck', 'goose'], ['duck','bend','crouch']]:
        print('{0} --> {1}'.format('-'.join(pair), ', '.join([word_sim for (word_sim, cos_sim) in self.word_embeddings.most_similar(positive=pair)])))

    print('WHICH WORD DOES NOT MATCH IN THE GROUP?')
    print(self.word_embeddings.doesnt_match(['apple','rock','banana','lemon']))

    print('IF MAN IS A KING, THEN WOMAN IS A ....')
    print(self.word_embeddings.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))

    print('PARIS IS TO FRANCE, WHAT ATHENS IS TO ?')
    print(self.word_embeddings.most_similar(positive=['athens', 'france'], negative=['paris'], topn=1))


if __name__ == '__main__':
    print('Reading training data')
    documents = EnglishNerParser().read_file(os.path.join(DATA_DIR, 'ner', 'eng.train.txt'))
    print('Create features')
    vectorizer = Vectorizer(word_embedding_path=os.path.join(DATA_DIR, 'embeddings', 'glove.6B.50d.w2v.txt'))
    max_length = 60
    x_words, x_pos, x_shapes = vectorizer.encode_features(documents, max_length)
    labels = vectorizer.encode_annotations(documents)
    print('Loaded {} data samples'.format(len(x_words)))

    print('Padding sequences')
    y_train = sequence.pad_sequences([np_utils.to_categorical(y_group, num_classes=len(vectorizer.labels)) for y_group in labels[:int(len(labels) * 0.8)]], maxlen=max_length)
    y_validation = sequence.pad_sequences([np_utils.to_categorical(y_group, num_classes=len(vectorizer.labels)) for y_group in labels[int(len(labels) * 0.8):]], maxlen=max_length)

    print('Build neural net...')
    model = Recurrent.build_tagging('bilstm_lstm', word_embeddings=vectorizer.word_embeddings,
                                    input_shape={'pos': (len(vectorizer.pos2index), 10),
                                                 'shape': (len(vectorizer.shape2index), 2)},
                                    out_shape=len(vectorizer.labels),
                                    units=100, dropout_rate=0.5)
    print('Train neural net...')
    trained_model_name = os.path.join(DATA_DIR, 'models', 'ner_weights.h5')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    saveBestModel = ModelCheckpoint(trained_model_name, monitor='val_loss', verbose=1, save_best_only=True,
                                    mode='auto')
    model.fit([x_words[:int(len(x_words) * 0.8)], x_pos[:int(len(x_pos) * 0.8)], x_shapes[:int(len(x_shapes) * 0.8)]],
              y_train,
              validation_data=([x_words[int(len(x_words) * 0.8):], x_pos[int(len(x_pos) * 0.8):], x_shapes[int(len(x_shapes) * 0.8):]], y_validation),
              batch_size=32,  epochs=10, callbacks=[saveBestModel, early_stopping])

    model.load_weights(trained_model_name)
    model.save(os.path.join(DATA_DIR, 'models', 'ner.h5'))

    print('================== Classification Report =======================')
    y_pred, y_dev = [], []
    for i in range(int(len(labels) * 0.8), len(labels) - 1):
        y_p = model.predict([x_words[i], x_pos[i], x_shapes[i]],
                                    batch_size=1, verbose=0)
        y_p = NeuralNetwork.probas_to_classes(y_p)
        y_pred.extend(y_p.flatten())
        y_dev.extend(labels[i].flatten())

    print(classification_report(y_dev, y_pred))
