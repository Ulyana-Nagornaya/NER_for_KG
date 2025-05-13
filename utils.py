from datasets import load_dataset
import numpy as np

def load_data():
    dataset = load_dataset("DFKI-SLT/conll04")
    df_train = dataset["train"].to_pandas()
    df_test = dataset["test"].to_pandas()
    df_val = dataset["validation"].to_pandas()
    return df_train, df_test, df_val

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

def prepare_data_for_bert( df_train, df_test, df_val):
    df_train['bio_tags'] = df_train.apply(lambda row: generate_bio_tags(row['tokens'], row['entities']), axis=1)
    df_val['bio_tags'] = df_val.apply(lambda row: generate_bio_tags(row['tokens'], row['entities']), axis=1)
    df_test['bio_tags'] = df_test.apply(lambda row: generate_bio_tags(row['tokens'], row['entities']), axis=1)

    df_train = df_train[['tokens', 'bio_tags']]
    df_val = df_val[['tokens', 'bio_tags']]
    df_test = df_test[['tokens', 'bio_tags']]

    df_train['tokens'] = df_train['tokens'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    df_val['tokens'] = df_val['tokens'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    df_test['tokens'] = df_test['tokens'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return df_train, df_test, df_val






