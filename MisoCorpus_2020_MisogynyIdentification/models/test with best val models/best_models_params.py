
models_dict = {
    ###########################################
    #misocorpus_val_lower_preprocess_without_url
    ###########################################


   'bertin-project/bertin-roberta-base-spanish1': {
       'train_file': './misocorpus_train_lower_preprocess_without_url.csv',
       'val_file': './misocorpus_val_lower_preprocess_without_url.csv',
       'test_file': './misocorpus_test_lower_preprocess_without_url.csv',
       'emb_size': 512,
       'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
       'learning_rate': 1.3568863399842198e-05,
       'weight_decay': 0.00017418618667477376,
       'num_train_epochs': 6,
       'batch_size': 32,
       'dir': './Bertin/misocorpus_train_lower_preprocess_without_url'
   },
   'dccuchile/bert-base-spanish-wwm-uncased1': {
       'train_file': './misocorpus_train_lower_preprocess_without_url.csv',
       'val_file': './misocorpus_val_lower_preprocess_without_url.csv',
       'test_file': './misocorpus_test_lower_preprocess_without_url.csv',
       'emb_size': 512,
       'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
       'learning_rate': 6.535216843778054e-06,
       'weight_decay': 9.475311305429342e-05,
       'num_train_epochs': 5,
       'batch_size': 16,
       'dir': './Beto_uncased/misocorpus_train_lower_preprocess_without_url'
   },

  'PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy1': {
      'train_file': './misocorpus_train_lower_preprocess_without_url.csv',
      'val_file': './misocorpus_val_lower_preprocess_without_url.csv',
      'test_file': './misocorpus_test_lower_preprocess_without_url.csv',
      'emb_size': 512,
      'tokenizer': 'PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy',
      'learning_rate': 1.4943105440819098e-05,
      'weight_decay': 0.0022548174649825318,
      'num_train_epochs': 3,
      'batch_size': 32,
      'dir': './Roberta_bne/misocorpus_train_lower_preprocess_without_url'
  },


   'pysentimiento/robertuito-base-uncased1': {
       'train_file': './misocorpus_train_lower_preprocess_without_url.csv',
       'val_file': './misocorpus_val_lower_preprocess_without_url.csv',
       'test_file': './misocorpus_test_lower_preprocess_without_url.csv',
       'emb_size': 128,
       'tokenizer': 'pysentimiento/robertuito-base-uncased',
       'learning_rate':  1.5620524928821222e-05,
       'weight_decay': 4.6851424993736176e-05,
       'num_train_epochs': 4,
       'batch_size': 16,
       'dir': './Robertuito_uncased/misocorpus_train_lower_preprocess_without_url'
   },

   './models/beto-mlm-savemodel1': {
       'train_file': './misocorpus_train_lower_preprocess_without_url.csv',
       'val_file': './misocorpus_val_lower_preprocess_without_url.csv',
       'test_file': './misocorpus_test_lower_preprocess_without_url.csv',
       'emb_size': 512,
       'tokenizer': './model-beto-mlm-adaptado-misoginia-4epoch',
       'learning_rate': 1.603151446324787e-05,
       'weight_decay': 0.0032962398685122576,
       'num_train_epochs': 5,
       'batch_size': 16,
       'dir': './Beto_4epoch_superset/misocorpus_train_lower_preprocess_without_url'
   },

  './models/robertuito2epoch-superset-mlm-savemodel1': {
      'train_file': './misocorpus_train_lower_preprocess_without_url.csv',
      'val_file': './misocorpus_val_lower_preprocess_without_url.csv',
      'test_file': './misocorpus_test_lower_preprocess_without_url.csv',
      'emb_size': 128,
      'tokenizer': './model-robertuito-mlm-adaptado-misoginia-2epoch',
      'learning_rate': 1.4095566782245004e-05,
      'weight_decay':  0.002883510303741726,
      'num_train_epochs': 5,
      'batch_size': 16,
      'dir': './Robertuito_2epoch_superset/misocorpus_train_lower_preprocess_without_url'
  },

  './models/robertuito-superset-mlm-savemodel1': {
      'train_file': './misocorpus_train_lower_preprocess_without_url.csv',
      'val_file': './misocorpus_val_lower_preprocess_without_url.csv',
      'test_file': './misocorpus_test_lower_preprocess_without_url.csv',
      'emb_size': 128,
      'tokenizer': './model-robertuito-mlm-adaptado-misoginia-4epoch',
      'learning_rate': 1.1191074332287145e-05,
      'weight_decay': 0.003106360183478765,
      'num_train_epochs': 5,
      'batch_size': 16,
      'dir': './Robertuito_4epoch_superset/misocorpus_train_lower_preprocess_without_url'
  },
   


   ###########################################
   #misocorpus_val_lower_preprocess_without_emojis_and_url
   ###########################################


   'bertin-project/bertin-roberta-base-spanish3': {
       'train_file': './misocorpus_train_lower_preprocess_without_emojis_and_url.csv',
       'val_file': './misocorpus_val_lower_preprocess_without_emojis_and_url.csv',
       'test_file': './misocorpus_test_lower_preprocess_without_emojis_and_url.csv',
       'emb_size': 512,
       'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
       'learning_rate': 1.5622452148057287e-05,
       'weight_decay':  0.00048729904915177115,
       'num_train_epochs': 6,
       'batch_size': 8,
       'dir': './Bertin/misocorpus_train_lower_preprocess_without_emojis_and_url'
   },
   
   'dccuchile/bert-base-spanish-wwm-uncased3': {
       'train_file': './misocorpus_train_lower_preprocess_without_emojis_and_url.csv',
       'val_file': './misocorpus_val_lower_preprocess_without_emojis_and_url.csv',
       'test_file': './misocorpus_test_lower_preprocess_without_emojis_and_url.csv',
       'emb_size': 512,
       'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
       'learning_rate':  1.991808397929952e-05,
       'weight_decay':  0.004756061731547124,
       'num_train_epochs': 6,
       'batch_size': 16,
       'dir': './Beto_uncased/misocorpus_train_lower_preprocess_without_emojis_and_url'
   },

   'PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy2': {
       'train_file': './misocorpus_train_lower_preprocess_without_emojis_and_url.csv',
       'val_file': './misocorpus_val_lower_preprocess_without_emojis_and_url.csv',
       'test_file': './misocorpus_test_lower_preprocess_without_emojis_and_url.csv',
       'emb_size': 512,
       'tokenizer': 'PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy',
       'learning_rate': 1.594387745247208e-05,
       'weight_decay': 4.9904473959824616e-05,
       'num_train_epochs': 4,
       'batch_size': 16,
       'dir': './Roberta_bne/misocorpus_train_lower_preprocess_without_emojis_and_url'
   },


   'pysentimiento/robertuito-base-uncased3': {
       'train_file': './misocorpus_train_lower_preprocess_without_emojis_and_url.csv',
       'val_file': './misocorpus_val_lower_preprocess_without_emojis_and_url.csv',
       'test_file': './misocorpus_test_lower_preprocess_without_emojis_and_url.csv',
       'emb_size': 128,
       'tokenizer': 'pysentimiento/robertuito-base-uncased',
       'learning_rate': 1.7493467489493463e-05,
       'weight_decay':  0.0001171797648331026,
       'num_train_epochs': 5,
       'batch_size': 32,
       'dir': './Robertuito_uncased/misocorpus_train_lower_preprocess_without_emojis_and_url'
   },

   './models/beto-mlm-savemodel3': {
       'train_file': './misocorpus_train_lower_preprocess_without_emojis_and_url.csv',
       'val_file': './misocorpus_val_lower_preprocess_without_emojis_and_url.csv',
       'test_file': './misocorpus_test_lower_preprocess_without_emojis_and_url.csv',
       'emb_size': 512,
       'tokenizer': './model-beto-mlm-adaptado-misoginia-4epoch',
       'learning_rate': 1.9184755844202326e-05,
       'weight_decay': 4.940022405124677e-05,
       'num_train_epochs': 3,
       'batch_size': 16,
       'dir': './Beto_4epoch_superset/misocorpus_train_lower_preprocess_without_emojis_and_url'
   },

   './models/robertuito2epoch-superset-mlm-savemodel2': {
       'train_file': './misocorpus_train_lower_preprocess_without_emojis_and_url.csv',
       'val_file': './misocorpus_val_lower_preprocess_without_emojis_and_url.csv',
       'test_file': './misocorpus_test_lower_preprocess_without_emojis_and_url.csv',
       'emb_size': 128,
       'tokenizer': './model-robertuito-mlm-adaptado-misoginia-2epoch',
       'learning_rate': 1.6332742780987696e-05,
       'weight_decay': 0.00013973843161165487,
       'num_train_epochs': 5,
       'batch_size': 16,
       'dir': './Robertuito_2epoch_superset/misocorpus_train_lower_preprocess_without_emojis_and_url'
   },

   './models/robertuito-superset-mlm-savemodel2': {
       'train_file': './misocorpus_train_lower_preprocess_without_emojis_and_url.csv',
       'val_file': './misocorpus_val_lower_preprocess_without_emojis_and_url.csv',
       'test_file': './misocorpus_test_lower_preprocess_without_emojis_and_url.csv',
       'emb_size': 128,
       'tokenizer': './model-robertuito-mlm-adaptado-misoginia-4epoch',
       'learning_rate': 1.3519972891817288e-05,
       'weight_decay': 0.00047608872934513436,
       'num_train_epochs': 5,
       'batch_size': 32,
       'dir': './Robertuito_4epoch_superset/misocorpus_train_lower_preprocess_without_emojis_and_url'
   },



   ###########################################
   #misocorpus_val_lower_preprocess_default
   ###########################################


   'bertin-project/bertin-roberta-base-spanish4': {
       'train_file': './misocorpus_train_lower_preprocess_default.csv',
       'val_file': './misocorpus_val_lower_preprocess_default.csv',
       'test_file': './misocorpus_test_lower_preprocess_default.csv',
       'emb_size': 512,
       'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
       'learning_rate':  1.3132556233139315e-05,
       'weight_decay': 0.0021208453843132424,
       'num_train_epochs': 6,
       'batch_size': 8,
       'dir': './Bertin/misocorpus_val_lower_preprocess_default'
   },
   
   'dccuchile/bert-base-spanish-wwm-uncased4': {
       'train_file': './misocorpus_train_lower_preprocess_default.csv',
       'val_file': './misocorpus_val_lower_preprocess_default.csv',
       'test_file': './misocorpus_test_lower_preprocess_default.csv',
       'emb_size': 512,
       'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
       'learning_rate': 9.9739108215677e-06,
       'weight_decay':  0.0005843120735331682,
       'num_train_epochs': 3,
       'batch_size': 32,
       'dir': './Beto_uncased/misocorpus_val_lower_preprocess_default'
   },

    'PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy4': {
       'train_file': './misocorpus_train_lower_preprocess_default.csv',
       'val_file': './misocorpus_val_lower_preprocess_default.csv',
       'test_file': './misocorpus_test_lower_preprocess_default.csv',
       'emb_size': 512,
       'tokenizer': 'PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy',
       'learning_rate': 6.796190367374683e-06,
       'weight_decay':  0.0006670477405782003,
       'num_train_epochs': 3,
       'batch_size': 16,
       'dir': './Roberta_bne/misocorpus_val_lower_preprocess_default'
   },


   'pysentimiento/robertuito-base-uncased4': {
       'train_file': './misocorpus_train_lower_preprocess_default.csv',
       'val_file': './misocorpus_val_lower_preprocess_default.csv',
       'test_file': './misocorpus_test_lower_preprocess_default.csv',
       'emb_size': 128,
       'tokenizer': 'pysentimiento/robertuito-base-uncased',
       'learning_rate': 7.493465424367135e-06,
       'weight_decay': 0.003097549720981271,
       'num_train_epochs': 4,
       'batch_size': 8,
       'dir': './Robertuito_uncased/misocorpus_val_lower_preprocess_default'
   },
   
   './models/beto-mlm-savemodel4': {
       'train_file': './misocorpus_train_lower_preprocess_default.csv',
       'val_file': './misocorpus_val_lower_preprocess_default.csv',
       'test_file': './misocorpus_test_lower_preprocess_default.csv',
       'emb_size': 512,
       'tokenizer': './model-beto-mlm-adaptado-misoginia-4epoch',
       'learning_rate': 1.7064936702896934e-05 ,
       'weight_decay': 0.002967767120719754,
       'num_train_epochs': 4,
       'batch_size': 8,
       'dir': './Beto_4epoch_superset/misocorpus_val_lower_preprocess_default'
   },

   './models/robertuito2epoch-superset-mlm-savemodel4': {
       'train_file': './misocorpus_train_lower_preprocess_default.csv',
       'val_file': './misocorpus_val_lower_preprocess_default.csv',
       'test_file': './misocorpus_test_lower_preprocess_default.csv',
       'emb_size': 128,
       'tokenizer': './model-robertuito-mlm-adaptado-misoginia-2epoch',
       'learning_rate': 1.6619089803858907e-05 ,
       'weight_decay':  0.0013682211182981093,
       'num_train_epochs': 5,
       'batch_size': 8,
       'dir': './Robertuito_2epoch_superset/misocorpus_val_lower_preprocess_default'
   },

   './models/robertuito-superset-mlm-savemodel4': {
       'train_file': './misocorpus_train_lower_preprocess_default.csv',
       'val_file': './misocorpus_val_lower_preprocess_default.csv',
       'test_file': './misocorpus_test_lower_preprocess_default.csv',
       'emb_size': 128,
       'tokenizer': './model-robertuito-mlm-adaptado-misoginia-4epoch',
       'learning_rate': 1.0548043469914118e-05,
       'weight_decay': 0.0025214197170066767,
       'num_train_epochs': 5,
       'batch_size': 16,
       'dir': './Robertuito_4epoch_superset/misocorpus_val_lower_preprocess_default'
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


# Constants
DATASETS_DELIMITERS = {
    'MISOCORPUS_2020': ',',
}
SEED_VALUE = 1
os.environ["WANDB_DISABLED"] = "true"

class CustomTrainer(Trainer):
    def log(self, logs, iterator_start_time=None):
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs, iterator_start_time)

class HateSpeechClassifier:
    def __init__(self, model_name, config, dataset_dict):
        self.model_name = model_name
        self.config = config
        self.dataset = dataset_dict
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['tokenizer'],
            model_max_length=config['emb_size']
        )

    def preprocess_text(self, examples):
        inputs = [str(text) for text in examples["tweet"]]
        
        tokenized = self.tokenizer(
            inputs,
            truncation=True,
            padding=False,
            max_length=self.config["emb_size"]
        )
        
        tokenized["labels"] = [int(l) for l in examples["label"]]
        
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
        print(f"F1-score:  {f1_score(labels, preds):.4f}")
        print("\nReporte completo:")
        print(classification_report(labels, preds))

        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='macro'),
            'recall': recall_score(labels, preds, average='macro'),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_micro': f1_score(labels, preds, average='micro'),
            'f1_score': f1_score(labels, preds)
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
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        model.config.id2label = {0: 'No hate_speech', 1: 'hate_speech'}
        model.config.label2id = {'No hate_speech': 0, 'hate_speech': 1}

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


def cargar_dataset(ruta):
    df = pd.read_csv(
        ruta,
        skiprows=1,
        header=None,
        names = ["tweet", "label"],
        delimiter=DATASETS_DELIMITERS['MISOCORPUS_2020']
    )
    return df


if __name__ == "__main__":
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    random.seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_VALUE)


    for model_name, config in models_dict.items():

        df_train = cargar_dataset(config['train_file'])
        df_val = cargar_dataset(config['val_file'])
        df_test = cargar_dataset(config['test_file'])

        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "validation": Dataset.from_pandas(df_val),
            "test": Dataset.from_pandas(df_test)
        })

        classifier = HateSpeechClassifier(config['tokenizer'], config, dataset_dict)
        classifier.train_and_predict()



   


