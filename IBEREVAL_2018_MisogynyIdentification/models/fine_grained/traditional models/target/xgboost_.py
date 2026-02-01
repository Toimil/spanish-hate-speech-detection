import sys
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

DATASETS_DELIMITERS = {
    'IBEREVAL_MISOGYNY_2018': ','
}
SEED_VALUE = 1

parser = argparse.ArgumentParser(description="Entrenamiento y evaluación de un modelo XGBoost para target")
parser.add_argument('--train_file', type=str, required=True, help='Ruta al archivo CSV de entrenamiento')
parser.add_argument('--test_file', type=str, required=True, help='Ruta al archivo CSV de test')
args = parser.parse_args()
if not args.train_file or not args.test_file:
    parser.print_help()
    sys.exit(1)


def cargar_dataset(ruta):
    df = pd.read_csv(
        ruta,
        delimiter=DATASETS_DELIMITERS['IBEREVAL_MISOGYNY_2018']
    )
    return df


df_train_full = cargar_dataset(args.train_file)
df_test = cargar_dataset(args.test_file)

X = df_train_full["text"]
y = df_train_full["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=SEED_VALUE, stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), use_idf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(df_test["text"])


print("\n-----------------------------------------------------")
print("\n-----------------------------------------------------")
print(f"Archivo de entrenamiento: {args.train_file}")
print(f"Archivo de test: {args.test_file}")
print("XGBoost (target)")
print("\n-----------------------------------------------------")
print("\n-----------------------------------------------------")


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}

num_classes = y_train.nunique()

xgb = XGBClassifier(
    objective='multi:softmax',
    eval_metric='mlogloss',
    num_class=num_classes,
    random_state=SEED_VALUE,
    use_label_encoder=False
)

grid = GridSearchCV(xgb, param_grid, scoring='f1_macro', cv=5, verbose=1, n_jobs=-1)
grid.fit(X_train_tfidf, y_train)
print("\n--- Mejor combinación de hiperparámetros encontrada ---")
print(grid.best_params_)
best_model = grid.best_estimator_

def evaluar_modelo(modelo, X, y, conjunto):
    y_pred = modelo.predict(X)
    print(f"\n--- Métricas en conjunto de {conjunto} ---")
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred, average='macro'):.4f}")
    print(f"Recall:    {recall_score(y, y_pred, average='macro'):.4f}")
    print(f"F1-macro:  {f1_score(y, y_pred, average='macro'):.4f}")
    print(f"F1-micro:  {f1_score(y, y_pred, average='micro'):.4f}")
    print(f"F1-score:  {f1_score(y, y_pred):.4f}")
    print("\nReporte completo:")
    print(classification_report(y, y_pred))

evaluar_modelo(best_model, X_val_tfidf, y_val, "validación")
evaluar_modelo(best_model, X_test_tfidf, df_test["target"], "test")
