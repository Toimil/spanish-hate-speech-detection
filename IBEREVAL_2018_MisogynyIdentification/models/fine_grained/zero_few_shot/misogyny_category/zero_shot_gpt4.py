import sys
import pandas as pd
import argparse
import openai
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import os
import time

# Configuración
DATASETS_DELIMITERS = {'IBEREVAL_MISOGYNY_2018': ','}
SEED_VALUE = 1
openai.api_key = ""
def cargar_dataset(ruta):
    return pd.read_csv(ruta, delimiter=DATASETS_DELIMITERS['IBEREVAL_MISOGYNY_2018'])

def construir_prompt(texto):
    return f"""
    Eres un modelo de clasificación de textos misóginos que asigna una categoría a un texto dado.
    
    Las categorías posibles son:

    - 'stereotype' : Una imagen o idea generalizada, pero fija y simplificada, de una mujer; descripción del atractivo físico de las mujeres y/o comparaciones con estándares estrechos.
    - 'dominance' : Afirmar la superioridad de los hombres sobre las mujeres para resaltar la desigualdad de género.
    - 'derailing' : Para justificar el abuso a la mujer, rechazando la responsabilidad masculina; un intento de interrumpir la conversación para redirigir las conversaciones de las mujeres hacia algo más cómodo para los hombres.
    - 'sexual_harassment' : Para describir acciones como insinuaciones sexuales, solicitudes de favores sexuales, acoso de naturaleza sexual; intención de afirmar físicamente el poder sobre las mujeres mediante amenazas de violencia.
    - 'discredit' : Insultando a las mujeres sin ninguna otra intención mayor.
    - '0' : Si el texto no es misógino


    Decide la categoría del siguiente texto. Responde únicamente con 'stereotype' , 'dominance', 'derailing', 'sexual_harassment', 'discredit' o '0'.

Texto: "{texto}"
¿Cuál es su categoría?: """

def zero_shot_clasificar(textos):
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

            if 'stereotype' in resultado or 'Stereotype' in resultado:
                pred = 'stereotype'
            elif 'dominance' in resultado or 'Dominance' in resultado:
                pred = 'dominance'
            elif 'derailing' in resultado or 'Derailing' in resultado:
                pred = 'derailing'
            elif 'sexual_harassment' in resultado or 'Sexual_harassment' in resultado or 'Sexual Harassment' in resultado or 'sexual harassment' in resultado:
                pred = 'sexual_harassment'
            elif 'discredit' in resultado or 'Discredit' in resultado:
                pred = 'discredit'
            else:
                pred = '0'
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
    print(f"F1-score:  {f1_score(y_true, y_pred, average='weighted'):.4f}")
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
    test_preds = zero_shot_clasificar(df_test["text"].tolist())
    evaluar(df_test["misogyny_category"], test_preds, "test")

    nombre_archivo = "predicciones_zero_shot_" + os.path.splitext(os.path.basename(args.test_file))[0] + ".txt" 
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        for pred in test_preds:
            f.write(f"{pred}\n")
    print("\nPredicciones guardadas en:", nombre_archivo)
