from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report


class CRFModel:
    def __init__(self):
        self.model = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=50,
            all_possible_transitions=True
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_true_flat = [tag for sent in y_test for tag in sent]
        y_pred_flat = [tag for sent in y_pred for tag in sent]
        report = classification_report(y_true_flat, y_pred_flat, output_dict=True)
        return report