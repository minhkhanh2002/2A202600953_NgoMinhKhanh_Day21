import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

EVAL_THRESHOLD = 0.70


def _build_model(params: dict):
    """
    Bonus 2: Chon thuat toan dua tren tham so model_type.

    Ho tro: random_forest, gradient_boosting, logistic_regression.
    Mac dinh la random_forest neu khong chi dinh.
    """
    # Tao ban sao de khong thay doi dict goc
    p = dict(params)
    model_type = p.pop("model_type", "random_forest")

    if model_type == "gradient_boosting":
        return model_type, GradientBoostingClassifier(**p, random_state=42)
    elif model_type == "logistic_regression":
        # LogisticRegression khong nhan cac tham so cua tree-based models
        return model_type, LogisticRegression(max_iter=1000, random_state=42)
    else:
        # Mac dinh: random_forest
        return model_type, RandomForestClassifier(**p, random_state=42)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho model.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # ---- Bonus 5: Canh bao lech lac du lieu ----
    label_dist = y_train.value_counts(normalize=True)
    label_dist_dict = {str(int(k)): round(float(v), 4) for k, v in label_dist.items()}
    for cls in [0, 1, 2]:
        ratio = label_dist.get(cls, 0)
        if ratio < 0.10:
            print(
                f"⚠️  CANH BAO: Lop {cls} chiem {ratio:.2%} < 10% tong mau!"
            )
    # ---- End Bonus 5 ----

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen model
        # Bonus 2: su dung _build_model de chon thuat toan
        model_type, model = _build_model(params)
        mlflow.log_param("model_type_resolved", model_type)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # ---- Bonus 3: Bao cao hieu suat tu dong ----
        cm = confusion_matrix(y_eval, preds, labels=[0, 1, 2])
        precision, recall, _, _ = precision_recall_fscore_support(
            y_eval, preds, labels=[0, 1, 2], zero_division=0
        )

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.txt", "w") as f:
            f.write("=== Bao Cao Hieu Suat Mo Hinh ===\n\n")
            f.write(f"Model type : {model_type}\n")
            f.write(f"Accuracy   : {acc:.4f}\n")
            f.write(f"F1 (weighted): {f1:.4f}\n\n")
            f.write("=== Confusion Matrix ===\n")
            f.write("        Pred_0  Pred_1  Pred_2\n")
            for i, row in enumerate(cm):
                f.write(f"True_{i}  {row[0]:>6}  {row[1]:>6}  {row[2]:>6}\n")
            f.write("\n=== Precision & Recall theo tung lop ===\n")
            for cls in [0, 1, 2]:
                f.write(
                    f"  Lop {cls}: precision={precision[cls]:.4f}, recall={recall[cls]:.4f}\n"
                )
        print("Bao cao hieu suat da luu tai outputs/report.txt")
        # ---- End Bonus 3 ----

        # TODO 8: Luu metrics ra file outputs/metrics.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        metrics_data = {"accuracy": acc, "f1_score": f1}
        # Bonus 5: ghi phan phoi nhan vao metrics.json
        metrics_data["label_distribution"] = label_dist_dict
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

        # TODO 9: Luu mo hinh ra file models/model.pkl
        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Tra ve acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
