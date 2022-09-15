import numpy as np

class loan_model():
    def __init__(self, model, pos_thresh):
        self.model = model
        self.pos_thresh = pos_thresh
    def fit(self, X, y):
        self.model.fit(X, y)
        return self.model
    def predict_proba(self, X):
        self.prob_pred_raw = self.model.predict_proba(X)
        self.prob_pred = self.prob_pred_raw - np.array(
            [.5-self.pos_thresh,
             self.pos_thresh-.5])
        return self.prob_pred
    def predict(self, X):
        self.y_pred = np.apply_along_axis(
            np.argmax,
            axis=1,
            arr=self.predict_proba(X))
        return self.y_pred