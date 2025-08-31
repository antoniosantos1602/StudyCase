import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

def _clean_features_array(X_like) -> np.ndarray:
    """Converte para float32 e remove NaN/Inf (substitui)."""
    X_arr = np.asarray(X_like, dtype=np.float32)
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e9, neginf=-1e9)
    return X_arr

def _assert_finite(name: str, X_arr: np.ndarray):
    if not np.isfinite(X_arr).all():
        raise ValueError(f"{name} contém NaN/Inf. Verifica o pré-processamento.")

# =========================
#  Modelo com hiperparâmetros do Optuna
# =========================
def build_model(input_size: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, 256),      # hidden1
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1192),            # dropout
        nn.Linear(256, 32),              # hidden2
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1192),            # dropout
        nn.Linear(32, num_classes)       # logits; CrossEntropy aplica Softmax internamente
    )

# =========================
#  Utils
# =========================
def _ensure_numeric_labels(y: pd.Series) -> np.ndarray:
    """Garante rótulos numéricos. Se vier 'Benign'/'Malicious', mapeia para 0/1."""
    if y.dtype == object:
        ys = y.astype(str).str.strip()
        if set(ys.unique()) <= {"Benign", "Malicious"}:
            return (ys != "Benign").astype(int).to_numpy()
        # multiclasse em string → ordinal estável
        classes_sorted = np.array(sorted(ys.unique()))
        mapping = {c: i for i, c in enumerate(classes_sorted)}
        return ys.map(mapping).to_numpy()
    return y.to_numpy()

def _make_class_weights(y_train_enc: np.ndarray, num_classes: int) -> torch.Tensor:
    classes = np.arange(num_classes)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_enc)
    return torch.tensor(cw, dtype=torch.float32)

# =========================
#  Treino + guardar
# =========================
def train_and_save_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_path: str,
    *,
    epochs: int = 100,
    batch_size: int = 128,        # atualizado
    lr: float = 0.0038397,        # atualizado
    patience: int = 10,
    val_frac: float = 0.2,
    seed: int = 123,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Scaler
    scaler = MinMaxScaler()
    X_clean = pd.DataFrame(
        _clean_features_array(X_train.values),
        columns=X_train.columns,
        index=X_train.index
    )
    _assert_finite("X_train (antes do scaler)", X_clean.to_numpy())

    X_scaled_df = pd.DataFrame(
        scaler.fit_transform(X_clean),
        columns=X_train.columns,
        index=X_train.index
    )
    _assert_finite("X_train escalado", X_scaled_df.to_numpy())
    X_scaled = X_scaled_df.to_numpy(dtype=np.float32)

    # Labels
    y_enc = _ensure_numeric_labels(y_train)
    num_classes = int(np.unique(y_enc).size)
    input_size = X_scaled.shape[1]

    # Split treino/val
    n = X_scaled.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_len = int(n * val_frac)
    val_idx = idx[:val_len]
    trn_idx = idx[val_len:]

    X_trn = torch.tensor(X_scaled[trn_idx], dtype=torch.float32)
    y_trn = torch.tensor(y_enc[trn_idx], dtype=torch.long)
    X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_enc[val_idx], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_trn, y_trn), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # Modelo / perda / otimizador
    DEVICE = torch.device("cpu")
    model = build_model(input_size, num_classes).to(DEVICE)
    weight_tensor = _make_class_weights(y_enc[trn_idx], num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Treino com early stopping
    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= max(len(trn_idx), 1)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = criterion(model(xb), yb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= max(len(val_idx), 1)

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")

        if va_loss < best_val_loss - 1e-6:
            best_val_loss = va_loss
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("⏹️ Early stopping (val_loss estabilizou).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump((model, scaler), model_path)
    print(f"[OK] Guardado: {model_path}")

    return model, scaler

# =========================
#  Inferência
# =========================
def predict(model: nn.Module, X: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    model.eval()
    X_clean = pd.DataFrame(
        _clean_features_array(X.values),
        columns=X.columns,
        index=X.index
    )
    _assert_finite("X (inference, antes do scaler)", X_clean.to_numpy())
    X_scaled = scaler.transform(X_clean)

    _assert_finite("X (inference, escalado)", X_scaled)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds

# =========================
#  Execução direta
# =========================
if __name__ == "__main__":
    train_path = "../StudyCase/cicids2017-original//original-cicids2017-train.csv"
    test_path  = "../StudyCase/cicids2017-original/original-cicids2017-test.csv"
    model_path = "../StudyCase/cicids2017-original/multi-class/artifacts/mlp_model_pytorch_multi.joblib"

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_test  = df_test.drop(columns=["label"])
    y_test  = df_test["label"].to_numpy()

    model, scaler = train_and_save_model(
        X_train, y_train, model_path,
        epochs=100, batch_size=128, lr=0.0038397, patience=10, val_frac=0.2, seed=123
    )

    model_loaded, scaler_loaded = joblib.load(model_path)
    y_pred = predict(model_loaded, X_test, scaler_loaded)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
