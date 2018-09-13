from keras.models import load_model, save_model
from .neural_network import NeuralNetwork
from keras.layers import concatenate
from keras.layers import Input, Embedding, Dropout, Bidirectional, LSTM, Dense, TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.models import Model
from .layers import Attention


class Recurrent(NeuralNetwork):
    def __init__(self, model):
        self._model = model

    @classmethod
    def build_tagging(cls, model_description: str, word_embeddings, input_shape: dict, out_shape: int, units=100, dropout_rate=0.5):

        word_input = Input(shape=(None, ), dtype='int32', name='word_input')

        weights = word_embeddings.syn0
        word_embeddings = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1],
                                    weights=[weights], name="word_embeddings_layer", trainable=False,
                                    mask_zero=True)(word_input)

        pos_input = Input(shape=(None, ), dtype='int32', name='pos_input')
        pos_embeddings = Embedding(input_shape['pos'][0], input_shape['pos'][1], name='pos_embeddings_layer',
                                   mask_zero=True)(pos_input)

        shape_input = Input(shape=(None, ), dtype='int32', name='shape_input')
        shape_embeddings = Embedding(input_shape['shape'][0], input_shape['shape'][1], name='shape_embeddings_layer',
                                     mask_zero=True)(shape_input)

        merged_input = concatenate([word_embeddings, pos_embeddings, shape_embeddings], axis=-1)

        previous = merged_input
        for layer in model_description.split('_'):
            if layer == 'bilstm':
                previous = Bidirectional(LSTM(units, activation='tanh', return_sequences=True))(previous)
            elif layer == 'lstm':
                previous = LSTM(units, activation='tanh', return_sequences=True)(previous)

        dropout_layer = Dropout(dropout_rate)(previous)
        output = TimeDistributed(Dense(out_shape, activation='softmax'))(dropout_layer)
        model = Model(inputs=[word_input, pos_input, shape_input], outputs=output)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return Recurrent(model)

    @classmethod
    def build_classification(cls, model_description: str, word_embeddings, input_shape: dict, out_shape: int, units=100,
              dropout_rate=0.5):

        word_input = Input(shape=(None,), dtype='int32', name='word_input')

        weights = word_embeddings.syn0
        word_embeddings = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1],
                                    weights=[weights], name="word_embeddings_layer", trainable=False,
                                    mask_zero=True)(word_input)

        pos_input = Input(shape=(None,), dtype='int32', name='pos_input')
        pos_embeddings = Embedding(input_shape['pos'][0], input_shape['pos'][1], name='pos_embeddings_layer',
                                   mask_zero=True)(pos_input)

        shape_input = Input(shape=(None,), dtype='int32', name='shape_input')
        shape_embeddings = Embedding(input_shape['shape'][0], input_shape['shape'][1], name='shape_embeddings_layer',
                                     mask_zero=True)(shape_input)

        merged_input = concatenate([word_embeddings, pos_embeddings, shape_embeddings], axis=-1)
        droped_merged_input = Dropout(dropout_rate)(merged_input)
        previous = droped_merged_input
        if model_description == 'bilstm_lstm':
            previous = Bidirectional(LSTM(units, activation='tanh', return_sequences=True))(previous)
            previous = LSTM(units, activation='tanh')(previous)
        elif model_description == 'bilstm_attention':
            previous = Bidirectional(LSTM(units, activation='tanh', return_sequences=True))(previous)
            previous = Attention()(previous)
        # elif model_description == 'cnn':
        #     droped_word_input = word_embeddings
        #     filter_lengths = [2, 3, 4]
        #     n_gram_convs = []
        #     for l in filter_lengths:
        #         conv_gram = Convolution1D(units, l, padding='same', activation='relu',
        #                                   name='{}_gram'.format(l))(droped_word_input)
        #         pool_gram = MaxPooling1D()(conv_gram)
        #         conv_out = Flatten()(pool_gram)
        #         n_gram_convs.append(conv_out)
        #     previous = concatenate(n_gram_convs, name="multi_conv_out")

        dropout_layer = Dropout(dropout_rate)(previous)
        output = Dense(out_shape, activation='softmax')(dropout_layer)
        model = Model(inputs=[word_input, pos_input, shape_input], outputs=output)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        return Recurrent(model)

    @classmethod
    def load(cls, filename):
        return Recurrent(load_model(filename))

    def save(self, filename):
        save_model(self._model, filename)
        self._model.save(filename)
