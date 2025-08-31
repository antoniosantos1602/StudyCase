"""CICIDS2017 Dataset Preprocessing (label numérica)"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ---------------- I/O ----------------

def load_dataset_list(load_path_list, delimiter=","):
    df_list = [load_dataset(p, delimiter) for p in load_path_list]
    return pd.concat(df_list, ignore_index=True)

def load_dataset(load_path, delimiter=","):
    filetype = load_path.rpartition(".")[2].lower()
    if filetype == "csv":
        return pd.read_csv(load_path, delimiter=delimiter, keep_default_na=True)
    raise ValueError("Unexpected filetype.")

def save_dataset(df, save_path, delimiter=","):
    filetype = save_path.rpartition(".")[2].lower()
    if filetype == "csv":
        df.to_csv(save_path, sep=delimiter, index=False, encoding="utf-8")
    else:
        raise ValueError("Unexpected filetype.")

# ------------- inspeção ---------------

def check_column_values(df):
    for col_name in df.columns:
        unique_cat = len(df[col_name].unique())
        print("Feature '{}' has {} unique values:".format(col_name, unique_cat))
        print(df[col_name].value_counts().sort_values(ascending=False).head(10), "\n")

def check_features_and_classes(features, classes, y_train=None, y_test=None):
    print("Number of features: {}\n".format(features.shape[0]))
    for i, f in enumerate(features):
        print("Feature {} -> {}".format(i, f))
    print("\nNumber of classes: {}\n".format(classes.shape[0]))
    for i, c in enumerate(classes):
        print("Class {} -> {}".format(i, c))
        if y_train is not None:
            print("Training set support: {}".format(np.count_nonzero(y_train == c)))
        if y_test is not None:
            print("Holdout set support: {}".format(np.count_nonzero(y_test == c)))
        print()

# -------------- PREPROCESS ------------

def preprocess_cicids2017(
    df,
    train_size=0,
    binary=False,
    check_initial=False,
    check_final=False,
    seed=None,
):
    """Preprocess the CICIDS2017 dataset (gera label numérica)."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    cols_to_drop = [
        "flow_id",
        "source_ip",
        "destination_ip",
        "timestamp",
        "fwd_header_length.1",
        "source_port",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if check_initial:
        print("\n----------------------------------------\nInitial Dataset\n")
        df.info()
        print("\n")
        check_column_values(df)

    # -------- label textual normalizada --------
    if binary:
        # 0 = Benign, 1 = Malicious
        y_text = pd.Series(
            ["Benign" if x in ("BENIGN", "Normal") else "Malicious"
             for x in df["label"].astype(str)],
            name="label_text",
            dtype=str,
        )
    else:
        def norm_lab(x: str) -> str:
            x = str(x)
            if x in ("BENIGN", "Normal"): return "Benign"
            m = {
                "SSH-Patator":"Bf_Patator_SSH",
                "FTP-Patator":"Bf_Patator_FTP",
                "DoS Hulk":"DoS_Hulk",
                "DoS GoldenEye":"DoS_GoldenEye",
                "DoS slowloris":"DoS_Slowloris",
                "DoS Slowhttptest":"DoS_Slowhttptest",
                "Heartbleed":"Heartbleed",
            }
            return m.get(x, x.strip())
        y_text = df["label"].astype(str).map(norm_lab)
        y_text.name = "label_text"

    X = df.drop("label", axis=1)

    # --- one-hot para destination_port (53,80,443,other) ---
    if "destination_port" in X.columns:
        svc_map = {53: "53", 80: "80", 443: "443"}
        X["destination_port_cat"] = X["destination_port"].map(svc_map).fillna("other").astype(str)
        X["destination_port_cat"] = pd.Categorical(X["destination_port_cat"], categories=["53","80","443","other"])
        X = pd.get_dummies(X, columns=["destination_port_cat"], prefix="destination_port", dtype=int)
        X = X.drop(columns=["destination_port"], errors="ignore")

    # --- one-hot para protocol (6,17,other) ---
    if "protocol" in X.columns:
        proto_map = {6: "6", 17: "17"}
        X["protocol_cat"] = X["protocol"].map(proto_map).fillna("other").astype(str)
        X["protocol_cat"] = pd.Categorical(X["protocol_cat"], categories=["6","17","other"])
        X = pd.get_dummies(X, columns=["protocol_cat"], prefix="protocol", dtype=int)
        X = X.drop(columns=["protocol"], errors="ignore")

    # garantir colunas OHE fixas
    for col in [
        "destination_port_53","destination_port_80","destination_port_443","destination_port_other",
        "protocol_6","protocol_17","protocol_other"
    ]:
        if col not in X.columns:
            X[col] = 0

    # imputação
    to_impute = ["flow_packets/s", "flow_bytes/s"]
    imp = SimpleImputer(missing_values=np.nan, strategy="median")
    row_benign = y_text[y_text == "Benign"].index
    if set(to_impute).issubset(X.columns) and len(row_benign) > 0:
        imp.fit(X.loc[row_benign, to_impute])
        X[to_impute] = imp.transform(X[to_impute])

    X.fillna(0, inplace=True)
    X.reset_index(drop=True, inplace=True)
    y_text.reset_index(drop=True, inplace=True)

    # -------- transformar label para NUMÉRICA --------
    if binary:
        # 0 = Benign, 1 = Malicious
        y_num = (y_text != "Benign").astype(int)
        class_names = np.array([0, 1])  # só para logs
    else:
        # IDs estáveis por ordem alfabética dos nomes normalizados
        classes_sorted = sorted(y_text.unique().tolist())
        cat = pd.Categorical(y_text, categories=classes_sorted)
        y_num = pd.Series(cat.codes, name="label", dtype=int)  # 0..K-1
        class_names = np.arange(len(classes_sorted))
        # log do mapeamento
        print("\nMapa de classes (nome -> id):")
        for i, name in enumerate(classes_sorted):
            print(f"{name} -> {i}")

    if check_final:
        features = X.columns.to_numpy(copy=True)
        classes = np.unique(y_num.to_numpy(copy=True))
        print("\n----------------------------------------\nFinal Dataset\n")

    if train_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_num, train_size=train_size, random_state=seed, shuffle=True, stratify=y_num
        )
        if check_final:
            check_features_and_classes(features, classes, y_train, y_test)
        # anexar label numérica de volta para guardar
        X_train = X_train.copy()
        X_train["label"] = y_train.values
        X_test = X_test.copy()
        X_test["label"] = y_test.values
        return X_train, X_test
    else:
        if check_final:
            check_features_and_classes(features, classes, y_num)
        X_out = X.copy()
        X_out["label"] = y_num.values
        return X_out, None

def start_preprocessing(
    prep_fn,
    load_path_list,
    save_path_train,
    save_path_test=None,
    load_delimiter=",",
    save_delimiter=",",
    **prep_fn_params
):
    # carregar
    if isinstance(load_path_list, list):
        df = load_dataset_list(load_path_list, load_delimiter)
    else:
        df = load_dataset(load_path_list, load_delimiter)

    # amostragem opcional
    sample_fraction = prep_fn_params.pop("sample_fraction", None)
    if sample_fraction:
        df = df.sample(frac=sample_fraction, random_state=prep_fn_params.get("seed", 123))

    df_train, df_test = prep_fn(df, **prep_fn_params)

    save_dataset(df_train, save_path_train, save_delimiter)
    if df_test is not None and save_path_test is not None:
        save_dataset(df_test, save_path_test, save_delimiter)

# ----------------- MAIN -----------------

if __name__ == "__main__":
    with open("../StudyCase/cicids2017-original/preprocessing-log.txt", "w") as log:
        sys.stdout = log

        load_path_list = [
            "/Users/antoniosantos171/Desktop/TrafficLabelling /Tuesday-WorkingHours.pcap_ISCX.csv",
            "/Users/antoniosantos171/Desktop/TrafficLabelling /Wednesday-workingHours.pcap_ISCX.csv",
        ]
        save_path_train = "../StudyCase/cicids2017-original/original-cicids2017-train.csv"
        save_path_test  = "../StudyCase/cicids2017-original/original-cicids2017-test.csv"

        prep_fn_params = {
            "train_size": 0.7,
            "binary": False,      # <<< binário agora: 0=Benign, 1=Malicious
            "check_initial": True,
            "check_final": True,
            "seed": 123,
            "sample_fraction": 0.05,
        }

        start_preprocessing(
            prep_fn=preprocess_cicids2017,
            load_path_list=load_path_list,
            save_path_train=save_path_train,
            save_path_test=save_path_test,
            load_delimiter=",",
            save_delimiter=",",
            **prep_fn_params
        )

        print("\nPreprocessing concluído (label numérica).")
