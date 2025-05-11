from datasets import load_dataset
import spacy

nlp = spacy.load("en_core_web_sm")

label2id = {
    "O": 0,
    "B-Peop": 1, "I-Peop": 2,
    "B-Org": 3, "I-Org": 4,
    "B-Loc": 5, "I-Loc": 6,
    "B-Other": 7, "I-Other": 8
}
id2label = {v: k for k, v in label2id.items()}

def load_data():
    dataset = load_dataset("DFKI-SLT/conll04")
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()
    return df_train, df_test

def add_pos_tags(row):
    tokens = row["tokens"]
    doc = nlp(" ".join(tokens))
    row["pos_tags"] = [token.pos_ for token in doc]
    return row

def generate_bio_tags(tokens, entities):
    bio_tags = ["O"] * len(tokens)
    for entity in entities:
        start = entity["start"]
        end = entity["end"]
        typ = entity["type"]
        if start < len(bio_tags):
            bio_tags[start] = f"B-{typ}"
            for i in range(start + 1, end):
                if i < len(bio_tags):
                    bio_tags[i] = f"I-{typ}"
    return bio_tags

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

