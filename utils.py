import pandas as pd
from datasets import load_dataset


def load_data():
    dataset = load_dataset("DFKI-SLT/conll04")
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()
    return df_train, df_test


def extract_features(sentence, index):
    word = sentence[index]
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }
    if index > 0:
        prev_word = sentence[index - 1]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:word.istitle()": prev_word.istitle(),
            "-1:word.isupper()": prev_word.isupper(),
        })
    else:
        features["BOS"] = True

    if index < len(sentence) - 1:
        next_word = sentence[index + 1]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:word.istitle()": next_word.istitle(),
            "+1:word.isupper()": next_word.isupper(),
        })
    else:
        features["EOS"] = True

    return features


def sent2features(sentence):
    return [extract_features(sentence, i) for i in range(len(sentence))]


def create_named_entities(entities, tokens):
    named_entities = []
    for entity in entities:
        start = entity['start']
        end = entity['end']
        entity_type = entity['type']
        entity_text = ' '.join(tokens[start:end])
        named_entities.append(f"{entity_text}({start}, {end}): {entity_type}")
    return named_entities


def generate_bio_tags(tokens, entities):
    bio_tags = ['O'] * len(tokens)
    for entity in entities:
        start = entity['start']
        end = entity['end']
        entity_type = entity['type']
        bio_tags[start] = f'B-{entity_type}'
        for i in range(start + 1, end):
            bio_tags[i] = f'I-{entity_type}'
    return bio_tags


def add_pos_tags(row):
    return row


def prepare_data(df):
    df['bio_tags'] = df.apply(lambda x: generate_bio_tags(x['tokens'], x['entities']), axis=1)
    df['tokens'] = df['tokens'].apply(lambda x: x.tolist() if isinstance(x, list) else x)
    X = df['tokens'].apply(sent2features).tolist()
    y = df['bio_tags'].tolist()
    return X, y