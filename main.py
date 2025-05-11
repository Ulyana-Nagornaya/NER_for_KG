import logging
from models.crf_ml import CRFModel
from models.crf_llm import BertCrfModel
from utils import load_data, prepare_data
import numpy as np

logging.basicConfig(
    filename='results.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_experiment(model, model_name, X_train, y_train, X_test, y_test):
    logging.info(f"[{model_name}] Starting training...")
    model.train(X_train, y_train)

    logging.info(f"[{model_name}] Starting evaluation...")
    metrics = model.evaluate(X_test, y_test)

    logging.info(f"[{model_name}] Results:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            logging.info(f"  {key}:")
            for k, v in value.items():
                logging.info(f"{k}: {v:.4f}")
        else:
            logging.info(f"{key}: {value:.4f}")

    logging.info(f"[{model_name}] Micro-F1 score: {metrics.get('accuracy', 0.0):.4f}")
    return metrics

def main():
    logging.info("Starting experiments: NER for Knowledge Graph Construction")

    logging.info("Loading CoNLL04 dataset...")
    df_train, df_test = load_data()

    logging.info("Preparing data for CRF model...")
    df_train, X_crf_train, y_crf_train = prepare_data(df_train)
    df_test, X_crf_test, y_crf_test = prepare_data(df_test)
    

    logging.info("Initializing CRF model...")
    crf_model = CRFModel()
    run_experiment(crf_model, "CRF", X_crf_train, y_crf_train, X_crf_test, y_crf_test)

    # logging.info("Preparing data for BERT+CRF model...")
    # df_train, X_crf_train, y_crf_train = prepare_data(df_train)
    # df_test, X_crf_test, y_crf_test = prepare_data(df_test)
    # df_val_bert, X_bert_val, y_bert_val = prepare_data(df_val)

    # logging.info("Preparing data for BERT+CRF model...")
    # X_bert_train = df_train["tokens"].tolist()
    # y_bert_train = df_train["bio_tags"].tolist()
    # X_bert_test = df_test["tokens"].tolist()
    # y_bert_test = df_test["bio_tags"].tolist()

    # logging.info("Initializing BERT+CRF model...")
    # bert_crf_model = BertCrfModel()
    # run_experiment(bert_crf_model, "BERT+CRF", X_bert_train, y_bert_train, X_bert_test, y_bert_test)

if __name__ == "__main__":
    main()