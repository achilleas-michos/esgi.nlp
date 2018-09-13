

class NeuralNetwork:
    def __init__(self):
        self._model = None

    def load_weights(self, *args, **kwargs):
        return self._model.load_weights(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self._model.fit(*args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        return self._model.fit_generator(*args, **kwargs)

    def predict_generator(self, *args, **kwargs):
        return self._model.predict_generator(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self._model.predict(*args, **kwargs)

    @staticmethod
    def probas_to_classes(proba):
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')