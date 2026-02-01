#IMPORTS 
import pandas as pd
import argparse
import os
from nltk.corpus import stopwords
import nltk
from pysentimiento.preprocessing import preprocess_tweet
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Constants
DATASETS_DELIMITERS = {
    'MISOCORPUS_2020': ',',
}
SEED_VALUE = 1

# FUNCIONES DE PREPROCESAMIENTO
def lower_preprocess_default(text):
    return preprocess_tweet(text, lang="es").lower()

def lower_preprocess_default_stopwords(text):
    return " ".join([word for word in preprocess_tweet(
        text, lang="es").lower().split() if word not in stop_words])



# DICCIONARIO DE VARIANTES
preprocess_variants = {
    "lower_preprocess_default": lower_preprocess_default,
    "lower_preprocess_default_stopwords": lower_preprocess_default_stopwords,
}

# CARGAR DATASET
def cargar_dataset(ruta):
    df = pd.read_csv(
        ruta,
        skiprows=1,
        header=None,
        names = ["tweet", "label"],
        delimiter=DATASETS_DELIMITERS['MISOCORPUS_2020']
    )
    return df


# GUARDAR VARIANTES
def guardar_variantes(df, base_output_name, tipo):
    for nombre, funcion in preprocess_variants.items():
        df_copy = df.copy()
        df_copy["tweet"] = df_copy["tweet"].apply(funcion)
        output_name = f"{base_output_name}_{tipo}_{nombre}.csv"
        df_copy.to_csv(output_name, index=False, sep=",")
        print(f"Guardado: {output_name}")

# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera CSVs con distintas variantes de preprocesamiento.")
    parser.add_argument("--train_file", required=True, help="Archivo de entrenamiento")
    parser.add_argument("--val_file", required=True, help="Archivo de validación")
    parser.add_argument("--test_file", required=True, help="Archivo de test")
    parser.add_argument("--output_prefix", default="misocorpus", help="Prefijo para nombres de archivos de salida")
    args = parser.parse_args()

    df_train = cargar_dataset(args.train_file)
    df_val = cargar_dataset(args.val_file)
    df_test = cargar_dataset(args.test_file)

    guardar_variantes(df_train, args.output_prefix, tipo="train")
    guardar_variantes(df_val, args.output_prefix, tipo="val")
    guardar_variantes(df_test, args.output_prefix, tipo="test")
