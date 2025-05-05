import logging
from models.crf_ml import CRFModel
from utils import load_data, prepare_data

logging.basicConfig(
    filename='results.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    logging.info("Loading CoNLL04 dataset...")
    df_train, df_test = load_data()

    logging.info("Data preprocessing...")
    X_train, y_train = prepare_data(df_train)
    X_test, y_test = prepare_data(df_test)

    model_name = "CRF"
    model = CRFModel()

    logging.info(f"[{model_name}] Starting training...")
    model.train(X_train, y_train)

    logging.info(f"[{model_name}] Starting evaluation...")
    metrics = model.evaluate(X_test, y_test)

    logging.info(f"[{model_name}] Results: {metrics}")
    return metrics


if __name__ == "__main__":
    main()