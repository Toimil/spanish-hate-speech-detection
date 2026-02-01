###########################################
###########################################
###########################################
###########################################


models_dict = {
    ###########################################
    #ibereval_train_lower_preprocess_without_emojis_and_url
    ###########################################


    'bertin-project/bertin-roberta-base-spanish1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'learning_rate': 1.5405114787159266e-05,
        'weight_decay': 0.00028345280109679926,
        'num_train_epochs': 5,
        'batch_size': 32,
        'dir': './Bertin/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    'dccuchile/bert-base-spanish-wwm-uncased1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'learning_rate': 4.215419480816439e-06,
        'weight_decay': 0.002021634488419174,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Beto_uncased/ibereval_train_lower_preprocess_without_emojis_and_url'
    },


    'pysentimiento/robertuito-base-uncased1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'learning_rate':  2.8316418005217568e-06,
        'weight_decay': 0.002552163771845794,
        'num_train_epochs': 3,
        'batch_size': 8,
        'dir': './Robertuito_uncased/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    './models/beto-mlm-savemodel1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': './model-beto-mlm-adaptado-misoginia-4epoch',
        'learning_rate': 8.38915149982555e-06,
        'weight_decay': 0.0003124098411187259,
        'num_train_epochs': 6,
        'batch_size': 32,
        'dir': './Beto_4epoch_superset/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    './models/robertuito2epoch-superset-mlm-savemodel1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './model-robertuito-mlm-adaptado-misoginia-2epoch',
        'learning_rate': 1.1420846339672208e-05,
        'weight_decay':   0.00017713382692391216,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Robertuito_2epoch_superset/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    './models/robertuito-superset-mlm-savemodel1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './model-robertuito-mlm-adaptado-misoginia-4epoch',
        'learning_rate': 1.5055367382279794e-06,
        'weight_decay': 0.0001054808744572524,
        'num_train_epochs': 4,
        'batch_size': 8,
        'dir': './Robertuito_4epoch_superset/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    'PlanTL-GOB-ES/roberta-base-bne1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy',
        'learning_rate': 1.1673938466480784e-05,
        'weight_decay':  0.00025489216539131024,
        'num_train_epochs': 6,
        'batch_size': 32,
        'dir': './Roberta_bne/ibereval_train_lower_preprocess_without_emojis_and_url'
    },
    


    ###########################################
    #ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url
    ###########################################


    'bertin-project/bertin-roberta-base-spanish3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'learning_rate': 9.371610247092188e-06,
        'weight_decay': 6.095083270763495e-05,
        'num_train_epochs': 5,
        'batch_size': 32,
        'dir': './Bertin/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },
        'dccuchile/bert-base-spanish-wwm-uncased3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'learning_rate':  1.4087808607551344e-05,
        'weight_decay':  4.202486346067173e-05,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Beto_uncased/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    'pysentimiento/robertuito-base-uncased3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'learning_rate': 1.4623149406491294e-05,
        'weight_decay': 0.005625972968472816,
        'num_train_epochs': 5,
        'batch_size': 32,
        'dir': './Robertuito_uncased/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    './models/beto-mlm-savemodel3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': './model-beto-mlm-adaptado-misoginia-4epoch',
        'learning_rate':  4.568266836530971e-06,
        'weight_decay': 6.773925010260788e-05,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Beto_4epoch_superset/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    './models/robertuito2epoch-superset-mlm-savemodel2': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './model-robertuito-mlm-adaptado-misoginia-2epoch',
        'learning_rate': 7.538396169186892e-06,
        'weight_decay':  4.133315330519081e-05,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Robertuito_2epoch_superset/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    './models/robertuito-superset-mlm-savemodel2': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './model-robertuito-mlm-adaptado-misoginia-4epoch',
        'learning_rate': 6.896312853168837e-06,
        'weight_decay': 0.006570569556088027,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Robertuito_4epoch_superset/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    'PlanTL-GOB-ES/roberta-base-bne3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy',
        'learning_rate': 1.6075131781348577e-05,
        'weight_decay': 0.0003576367947621834,
        'num_train_epochs': 3,
        'batch_size': 8,
        'dir': './Roberta_bne/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },




    ###########################################
    #ibereval_train_lower_preprocess_default
    ###########################################


    'bertin-project/bertin-roberta-base-spanish4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'learning_rate':  5.987140800221582e-06,
        'weight_decay': 0.004918330492867055,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Bertin/ibereval_train_lower_preprocess_default'
    },
    
    'dccuchile/bert-base-spanish-wwm-uncased4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'learning_rate': 9.355632259889727e-06,
        'weight_decay':  0.0001755064259540122,
        'num_train_epochs': 4,
        'batch_size': 16,
        'dir': './Beto_uncased/ibereval_train_lower_preprocess_default'
    },

    'pysentimiento/robertuito-base-uncased4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'learning_rate': 1.4204122812082982e-05,
        'weight_decay': 0.0005232184251993722,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Robertuito_uncased/ibereval_train_lower_preprocess_default'
    },
    
    './models/beto-mlm-savemodel4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 512,
        'tokenizer': './model-beto-mlm-adaptado-misoginia-4epoch',
        'learning_rate':  1.2986702926236104e-05,
        'weight_decay': 0.00616906133476823,
        'num_train_epochs': 5,
        'batch_size': 16,
        'dir': './Beto_4epoch_superset/ibereval_train_lower_preprocess_default'
    },

    './models/robertuito2epoch-superset-mlm-savemodel4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 128,
        'tokenizer': './model-robertuito-mlm-adaptado-misoginia-2epoch',
        'learning_rate':  1.1437210749698457e-05,
        'weight_decay':  9.218412833636681e-05,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Robertuito_2epoch_superset/ibereval_train_lower_preprocess_default'
    },

    './models/robertuito-superset-mlm-savemodel4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 128,
        'tokenizer': './model-robertuito-mlm-adaptado-misoginia-4epoch',
        'learning_rate': 1.7182768997686326e-05,
        'weight_decay':  0.0010740894194548923,
        'num_train_epochs': 4,
        'batch_size': 8,
        'dir': './Robertuito_4epoch_superset/ibereval_train_lower_preprocess_default'
    },

    'PlanTL-GOB-ES/roberta-base-bne4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 512,
        'tokenizer': 'PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy',
        'learning_rate': 1.5334170066593054e-05,
        'weight_decay': 0.003173573222608539,
        'num_train_epochs': 5,
        'batch_size': 16,
        'dir': './Roberta_bne/ibereval_train_lower_preprocess_default'
    },

}



###########################################
###########################################
###########################################
###########################################







import gc
import os
import argparse
import pandas as pd
import numpy as np
import json
import torch
import random
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

DATASETS_DELIMITERS = {'IBEREVAL_MISOGYNY_2018': ','}
SEED_VALUE = 1
os.environ["WANDB_DISABLED"] = "true"

class CustomTrainer(Trainer):
    def log(self, logs, iterator_start_time=None):
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs, iterator_start_time)

class HateSpeechClassifier:
    def __init__(self, model_name, config, dataset_dict, y_col):
        self.model_name = model_name
        self.config = config
        self.dataset = dataset_dict
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['tokenizer'],
            model_max_length=config['emb_size']
        )
        self.y_col = y_col
        self.labels = ['0', 'active', 'passive']
        self.label2id = {'0': 0, 'active': 1, 'passive': 2}
        self.id2label = {0: '0', 1: 'active', 2: 'passive'}


    def preprocess_text(self, examples):
        inputs = [str(text) for text in examples["text"]]
        
        tokenized = self.tokenizer(
            inputs,
            truncation=True,
            padding=False,
            max_length=self.config["emb_size"]
        )
        
        tokenized["labels"] = [int(self.label2id[l]) for l in examples["target"]]
        
        return tokenized

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        print("\n--- Métricas -------")
        print("Model:", self.model_name)
        print("\n--- Métricas -------")
        print(f"Accuracy:  {accuracy_score(labels, preds):.4f}")
        print(f"Precision: {precision_score(labels, preds, average='macro'):.4f}")
        print(f"Recall:    {recall_score(labels, preds, average='macro'):.4f}")
        print(f"F1-macro:  {f1_score(labels, preds, average='macro'):.4f}")
        print(f"F1-micro:  {f1_score(labels, preds, average='micro'):.4f}")
        print(f"F1-score:  {f1_score(labels, preds, average='weighted'):.4f}")
        print("\nReporte completo:")
        print(classification_report(labels, preds))

        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='macro'),
            'recall': recall_score(labels, preds, average='macro'),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_micro': f1_score(labels, preds, average='micro'),
            'f1_score': f1_score(labels, preds, average='weighted')
        }

    def predict_and_save(self, trainer, dataset, filename, conjunto="test"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        predictions = trainer.predict(dataset)
        preds = np.argmax(predictions.predictions, axis=1)

        df = pd.DataFrame({'predictions': preds})
        df.to_csv(filename, index=False)
        print(f" Predicciones guardadas en {filename}")

        metrics = self.compute_metrics((predictions.predictions, dataset["labels"]))
        print(f" Métricas calculadas para {conjunto}")

        metrics_filename = filename.replace(".csv", "_metrics.json")
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f" Métricas guardadas en {metrics_filename}")

        return preds, dataset["labels"]

    def train_and_predict(self):
        num_labels = len(self.labels)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        
        model.config.id2label = self.id2label
        model.config.label2id = self.label2id

        tokenized_data = self.dataset.map(
            self.preprocess_text, 
            batched=True,
            remove_columns=self.dataset["train"].column_names 
        )
        tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=self.config['dir'],
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            num_train_epochs= self.config['num_train_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            eval_strategy ="epoch",
            save_strategy="no",
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            seed=SEED_VALUE
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

        # Validacion
        self.predict_and_save(trainer, tokenized_data['validation'],
                      os.path.join(self.config['dir'], "predictions_validation.csv"),
                      conjunto="validacion")
        # Test
        self.predict_and_save(trainer, tokenized_data['test'],
                      os.path.join(self.config['dir'], "predictions_test.csv"),
                      conjunto="test")
        

         # ======= LIMPIEZA COMPLETA DE MEMORIA POR TRIAL =======
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            del model
            del trainer

            with torch.no_grad():
                torch.cuda.empty_cache()
                gc.collect()

            print(f"[INFO] Memoria liberada correctamente")

        except Exception as e:
            print(f"[WARNING] Error liberando memoria : {e}")
        # ======================================================


def cargar_dataset(ruta):
    df = pd.read_csv(
        ruta,
        delimiter=DATASETS_DELIMITERS['IBEREVAL_MISOGYNY_2018']
    )
    return df


if __name__ == "__main__":
    random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_VALUE)


    for model_name, config in models_dict.items():

        df_train_full = cargar_dataset(config['train_file'])
        X_train, X_val, y_train, y_val = train_test_split(
            df_train_full["text"],
            df_train_full["target"],
            test_size=0.25,
            random_state=SEED_VALUE,
            stratify=df_train_full["target"]
        )

        df_train = pd.DataFrame({"text": X_train, "target": y_train}).reset_index(drop=True)
        df_val = pd.DataFrame({"text": X_val, "target": y_val}).reset_index(drop=True)

        df_test_full = cargar_dataset(config['test_file'])
        df_test = df_test_full[["text", "target"]].reset_index(drop=True)

        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "validation": Dataset.from_pandas(df_val),
            "test": Dataset.from_pandas(df_test)
        })

        classifier = HateSpeechClassifier(config['tokenizer'], config, dataset_dict, y_col="target")
        classifier.train_and_predict()



   


