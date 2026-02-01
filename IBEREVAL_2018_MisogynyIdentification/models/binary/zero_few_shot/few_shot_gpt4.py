import sys
import pandas as pd
import argparse
import openai
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import os
import time

DATASETS_DELIMITERS = {'IBEREVAL_MISOGYNY_2018': ','}
SEED_VALUE = 1
openai.api_key = ""
def cargar_dataset(ruta):
    return pd.read_csv(ruta, delimiter=DATASETS_DELIMITERS['IBEREVAL_MISOGYNY_2018'])

def construir_prompt(texto):
    return f"""Decide si el siguiente texto es misógino o no. Responde únicamente con 'sí' o 'no'.
Algunos ejemplos:
Texto: "las mujeres no deberían trabajar, su lugar está en casa." ¿Es misógino?: sí   
Texto: "que hijo de puta, te ha tocado la loteria" ¿Es misógino?: no 
Texto: "las mujeres son todas unas putas, no sirven para nada" ¿Es misógino?: sí
Texto: "tengo una perra muy bonita en casa, de raza labrador" ¿Es misógino?: no

Ahora decide si el siguiente texto es misógino o no. Responde únicamente con 'sí' o 'no'.
Texto: "{texto}"
¿Es misógino?: """

def few_shot_clasificar(textos):
    respuestas = []
    for texto in textos:
        prompt = construir_prompt(texto)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            #print(f"Texto: {texto} -> Respuesta: {response['choices'][0]['message']['content'].strip().lower()}\n")
            resultado = response['choices'][0]['message']['content'].strip().lower()
            pred = 1 if "sí" in resultado or "si" in resultado else 0
            #print(f"Predicción: {pred} para texto: {texto[:50]}...")
        except Exception as e:
            print(f"Error con texto: {texto[:50]}... -> {e}")
            pred = 0
        respuestas.append(pred)
        time.sleep(0.5)  # para evitar rate limits
    return respuestas

def evaluar(y_true, y_pred, conjunto):
    print(f"\n--- Métricas en conjunto de {conjunto} ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1-macro:  {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1-micro:  {f1_score(y_true, y_pred, average='micro'):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred):.4f}")
    print("\nReporte completo:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clasificación zero-shot con GPT.")
    parser.add_argument('--test_file', type=str, required=True, help='Ruta al archivo CSV de test')
    args = parser.parse_args()
    if not args.test_file:
        parser.print_help()
        sys.exit(1)


    df_test = cargar_dataset(args.test_file)

    print("\n--------------------------------------------------")
    print(f"Archivo de test cargado: {args.test_file}")
    print(f"\nEvaluando conjunto de test ({len(df_test)} muestras)...")
    test_preds = few_shot_clasificar(df_test["text"].tolist())
    evaluar(df_test["misogynous"], test_preds, "test")
    
    nombre_archivo = "predicciones_few_shot_" + os.path.splitext(os.path.basename(args.test_file))[0] + ".txt" 
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        for pred in test_preds:
            f.write(f"{pred}\n")
    print("\nPredicciones guardadas en", nombre_archivo)