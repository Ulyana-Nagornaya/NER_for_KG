from sklearn_crfsuite import CRF
from sklearn.metrics import classification_report
from utils import generate_bio_tags
import spacy

nlp = spacy.load("en_core_web_sm")

def add_pos_tags(row):
    tokens = row["tokens"]
    doc = nlp(" ".join(tokens))
    row["pos_tags"] = [token.pos_ for token in doc]
    return row

def get_char_ngrams(token, min_n=2, max_n=5):
    ngrams = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(token) - n + 1):
            ngrams.add(token[i:i + n])
    return ngrams

def extract_features(df_row, i):
    tokens = df_row["tokens"]
    pos_tags = df_row["pos_tags"]
    token = tokens[i]
    pos = pos_tags[i]
    features = {
        "bias": 1.0,
        "pos": pos,
        "word.lower()": token.lower(),
        "is_capitalized": token[0].isupper() if len(token) > 0 else False,
        "is_all_caps": token.isupper(),
        "word.is_punctuation": token in [".", ",", ";", "?", "!"],
        "prefix4": token[:4] if len(token) >= 4 else token,
        "suffix4": token[-4:] if len(token) >= 4 else token,
    }
    if i > 0:
        prev_token = tokens[i - 1]
        prev_pos = pos_tags[i - 1]
        features.update({
            "-1:word.lower()": prev_token.lower(),
            "-1:pos": prev_pos if len(prev_pos) >= 2 else prev_pos,
        })
    else:
        features["BOS"] = True
    if i < len(tokens) - 1:
        next_token = tokens[i + 1]
        next_pos = pos_tags[i + 1]
        features.update({
            "+1:word.lower()": next_token.lower(),
            "+1:pos": next_pos if len(next_pos) >= 2 else next_pos,
        })
    else:
        features["EOS"] = True
    return features

def sent2features(df_row):
    return [extract_features(df_row, i) for i in range(len(df_row["tokens"]))]

def sent2labels(df_row):
    return df_row["bio_tags"]

def prepare_data(df):
    df = df.apply(add_pos_tags, axis=1)
    df["bio_tags"] = df.apply(lambda row: generate_bio_tags(row["tokens"], row["entities"]), axis=1)
    X = df.apply(sent2features, axis=1).tolist()
    y = df.apply(sent2labels, axis=1).tolist()
    return  df, X, y
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