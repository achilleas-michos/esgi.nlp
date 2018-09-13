import os
from data import DATA_DIR
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from projects.neural_networks import NeuralNetwork, Recurrent
from projects.document_model import AmazonReviewParser
from projects.amazon_reviews import Vectorizer



def plot_sequence_length_distribution(docs):
    bins = np.linspace(0, 100, 5)
    plt.figure()
    plt.hist([len(doc.get_sentence_tokens()) for doc in docs], bins=bins)
    plt.title('CONLL-NER-2002')
    plt.xlabel('Sequence Length')
    plt.ylabel('Occurrences')
    plt.savefig('{0}_histogram.png'.format('CONLL-NER-2002'))
    print('FIGURE SAVED')


if __name__ == '__main__':
    vectorizer = Vectorizer(word_embedding_path=os.path.join(DATA_DIR, 'embeddings', 'glove.6B.50d.w2v.txt'))

    print('Reading training data')
    documents = AmazonReviewParser().read_file(os.path.join(DATA_DIR, 'reviews', 'Automotive_5_train.json'))
    # plot_sequence_length_distribution(documents)
    print('Create features')
    features = vectorizer.encode_features(documents)
    labels = vectorizer.encode_annotations(documents)
    print('Loaded {} data samples'.format(len(features)))

    print('Padding sequences')
    max_length = 60
    x_train, x_validation, x_validation_unpadded = {}, {}, {}
    for key in features:
        x_train[key] = sequence.pad_sequences(features[key][:int(len(features[key]) * 0.8)], maxlen=max_length, dtype=np.int32)
        x_validation[key] = sequence.pad_sequences(features[key][int(len(features[key]) * 0.8):], maxlen=max_length, dtype=np.int32)
        x_validation_unpadded[key] = features[key][int(len(features[key]) * 0.8):]
    y_train = np_utils.to_categorical(labels[:int(len(labels) * 0.8)], num_classes=len(vectorizer.labels))
    y_validation = np_utils.to_categorical(labels[int(len(labels) * 0.8):], num_classes=len(vectorizer.labels))

    print('Build neural net...')
    model = Recurrent.build_classification('bilstm_lstm', word_embeddings=vectorizer.word_embeddings,
                            input_shape={'pos': (len(vectorizer.pos2index), 10),
                                         'shape': (len(vectorizer.shape2index), 2),
                                         'max_length': max_length},
                            out_shape=len(vectorizer.labels),
                            units=100, dropout_rate=0.5)
    print('Train neural net...')
    trained_model_name = os.path.join(DATA_DIR, 'models', 'reviews_weights.h5')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    saveBestModel = ModelCheckpoint(trained_model_name, monitor='val_loss', verbose=1, save_best_only=True,
                                    mode='auto')
    model.fit([x_train['words'], x_train['pos'], x_train['shape']], y_train,
              validation_data=([x_validation['words'], x_validation['pos'], x_validation['shape']], y_validation),
              batch_size=32,  epochs=1, callbacks=[saveBestModel, early_stopping])

    model.load_weights(trained_model_name)
    model.save(os.path.join(DATA_DIR, 'models', 'reviews.h5'))

    print('================== Classification Report =======================')
    y_pred, y_dev = [], []
    y_true = NeuralNetwork.probas_to_classes(y_validation)
    for i, label in enumerate(y_true):
        y_p = model.predict([np.asarray([x_validation_unpadded['words'][i]]),
                             np.asarray([x_validation_unpadded['pos'][i]]),
                             np.asarray([x_validation_unpadded['shape'][i]])])
        y_pred.extend(NeuralNetwork.probas_to_classes(y_p))
    print(classification_report(y_true, y_pred))
