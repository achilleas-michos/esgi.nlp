import os
from data import DATA_DIR
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint, EarlyStopping
from projects.neural_networks import Recurrent
from projects.document_model import EnglishPosParser
from projects.pos_en import Vectorizer


if __name__ == '__main__':
    vectorizer = Vectorizer(word_embedding_path=os.path.join(DATA_DIR, 'embeddings', 'glove.6B.50d.w2v.txt'))

    print('Reading training data')
    documents = EnglishPosParser().read_file(os.path.join(DATA_DIR, 'ner', 'eng.train.txt'))
    documents = documents[:10]
    print('Create features')
    features = vectorizer.encode_features(documents)
    labels = vectorizer.encode_annotations(documents)
    print('Loaded {} data samples'.format(len(features)))

    print('Padding sequences')
    max_length = 60
    x_train, x_validation = {}, {}
    for key in features:
        x_train[key] = sequence.pad_sequences(features[key][:int(len(features[key]) * 0.8)], maxlen=max_length)
        x_validation[key] = sequence.pad_sequences(features[key][int(len(features[key]) * 0.8):], maxlen=max_length)

    y_train = sequence.pad_sequences([np_utils.to_categorical(y_group, num_classes=len(vectorizer.labels)) for y_group in labels[:int(len(labels) * 0.8)]], maxlen=max_length)
    y_validation = sequence.pad_sequences([np_utils.to_categorical(y_group, num_classes=len(vectorizer.labels)) for y_group in labels[int(len(labels) * 0.8):]], maxlen=max_length)

    print('Build neural net...')
    model = Recurrent.build_tagging('bilstm_lstm', word_embeddings=vectorizer.word_embeddings,
                            input_shape={'pos': (len(vectorizer.pos2index), 10), 'shape': (len(vectorizer.shape2index), 2)},
                            out_shape=len(vectorizer.labels),
                            units=100, dropout_rate=0.5)
    print('Train neural net...')
    trained_model_name = os.path.join(DATA_DIR, 'models', 'ner.hdf5')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    saveBestModel = ModelCheckpoint(trained_model_name, monitor='val_loss', verbose=1, save_best_only=True,
                                    mode='auto')
    model.fit([x_train['words'], x_train['shape']], y_train,
              validation_data=([x_validation['words'], x_validation['shape']], y_validation),
              batch_size=32,  epochs=3, callbacks=[saveBestModel, early_stopping])

    model = Recurrent.load(trained_model_name)

    print('================== Classification Report =======================')
    y_pred, y_dev = [], []
    for i in range(int(len(labels) * 0.8), len(labels) - 1):
        y_p = model.predict_classes([features['words'][i], features['shape'][i]],
                                    batch_size=1, verbose=0)
        y_pred.extend(y_p.flatten())
        y_dev.extend(labels[i].flatten())

    print(classification_report(y_dev, y_pred))
