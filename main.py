import argparse
import logging
from models.crf_ml import CRFModel, prepare_data
from models.crf_llm import tokenizer, NERDataset, label2id, train_model, BertCrf_model, evaluate_model, get_classification_report, id2label
from utils import load_data, prepare_data_for_bert
from torch.utils.data import DataLoader

logging.basicConfig(
    filename='results.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(description="Train NER models for Knowledge Graph Construction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--crf", action="store_true", help="Train classic CRF model")
    group.add_argument("--bert-crf", action="store_true", help="Train BERT+CRF model")
    args = parser.parse_args()

    logging.info("Starting experiments: NER for Knowledge Graph Construction")

    if args.crf:
        logging.info("[CRF] Starting CRF model pipeline...")
        
        logging.info("[CRF] Loading dataset...")
        df_train, df_test, df_val = load_data()
        
        logging.info("[CRF] Preparing data...")
        _, X_train, y_train = prepare_data(df_train)
        _, X_test, y_test = prepare_data(df_test)
        
        logging.info("[CRF] Starting training...")
        model = CRFModel()
        model.train(X_train, y_train)
        
        logging.info("[CRF] Starting evaluation...")
        metrics = model.evaluate(X_test, y_test)
        
        logging.info("[CRF] Evaluation Results:")
        for key, value in metrics.items():
            if isinstance(value, dict):
                logging.info(f"  {key}:")
                for k, v in value.items():
                    logging.info(f"{k}: {v:.4f}")
            else:
                logging.info(f"{key}: {value:.4f}")
        logging.info(f"[CRF] Micro-F1 score: {metrics.get('accuracy', 0.0):.4f}")

    elif args.bert_crf:
        logging.info("[BERT+CRF] Starting BERT+CRF model pipeline...")
        
        logging.info("[BERT+CRF] Loading dataset...")
        df_train, df_test, df_val = load_data()
        
        logging.info("[BERT+CRF] Preparing data...")
        df_train, df_test, df_val = prepare_data_for_bert(df_train, df_test, df_val)
        
        logging.info("[BERT+CRF] Starting training...")
        train_dataset = NERDataset(df_train, tokenizer, label2id)
        train_loader = DataLoader(train_dataset, batch_size=17, shuffle=True)

        val_dataset = NERDataset(df_val, tokenizer, label2id)
        val_loader = DataLoader(val_dataset, batch_size=17, shuffle=False)

        train_model(BertCrf_model, train_loader, val_loader, epochs=10, patience=3)
            
        logging.info("[BERT+CRF] Starting evaluation...")
        test_dataset = NERDataset(df_test, tokenizer, label2id)
        test_loader = DataLoader(test_dataset, batch_size=17, shuffle=False)
        test_loss, test_preds, test_labels = evaluate_model(BertCrf_model, test_loader)
        
        logging.info("[BERT+CRF] Evaluation Results:")
        report = get_classification_report(test_preds, test_labels, id2label)
        for key, value in report.items():
            if isinstance(value, dict):
                logging.info(f"{key}:")
                for k, v in value.items():
                    logging.info(f"{k}: {v:.4f}")
            else:
                logging.info(f"{key}: {value:.4f}")
        logging.info(f"[BERT+CRF] Micro-F1 score: {report.get('accuracy', 0.0):.4f}")

if __name__ == "__main__":
    main()