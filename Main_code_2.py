import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ======================
# CONFIG
# ======================
DATA_PATH = r"C:/Users/Lenovo/Desktop/Final data"
WINDOW_SIZE = 20

X = []
y = []
raw_windows = []   # store real signal for plotting

# ======================
# LOAD + CLEAN (STRICT)
# ======================
def load_clean_csv(path):
    try:
        df = pd.read_csv(path, header=None, encoding="latin1", engine="python")

        if df.shape[1] == 1:
            col = df[0].astype(str).str.strip()
            df = col.str.split(",", expand=True)

            if df.shape[1] == 1:
                df = col.str.split(r"\s+", expand=True)

        if df.shape[1] < 3:
            print(" Bad format:", path)
            return None

        df = df.iloc[:, :3]
        df.columns = ["x", "y", "z"]

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        if len(df) < WINDOW_SIZE:
            print("❌ Too small:", path)
            return None

        return df

    except Exception as e:
        print(" Error:", path, e)
        return None


# ======================
# STRICT LABEL FUNCTION
# ======================
def get_label(label):
    label = label.lower()

    if "human" in label or "ha" in label:
        return "human_high" if "high" in label else "human_mid"

    elif "cooler" in label:
        return "cooler_high" if "high" in label else "cooler_mid"

    elif "mixer" in label:
        return "mixer_high" if "high" in label else "mixer_mid"

    else:
        return None



# LOAD DATA

for folder_name in os.listdir(DATA_PATH):
    folder = os.path.join(DATA_PATH, folder_name)

    if not os.path.isdir(folder):
        continue

    label_clean = get_label(folder_name)

    if label_clean is None:
        print("Unknown label:", folder_name)
        continue

    print("Reading:", label_clean)

    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(folder, file)
        df = load_clean_csv(path)

        if df is None:
            continue

        # ======================
        # SIGNAL PROCESSING
        # ======================
        df["mag"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
        df["mag"] = df["mag"] - df["mag"].mean()

        for i in range(0, len(df) - WINDOW_SIZE, WINDOW_SIZE):
            window = df["mag"].iloc[i:i+WINDOW_SIZE].values

            raw_windows.append(window)  #  real signal stored

            features = [
                np.mean(window),
                np.std(window),
                np.max(window),
                np.min(window),
                np.sqrt(np.mean(window**2)),
                np.percentile(window, 25),
                np.percentile(window, 75),
                np.median(window),
                np.var(window)
            ]

            fft_vals = np.abs(np.fft.fft(window))
            features += [
                np.mean(fft_vals),
                np.max(fft_vals),
                np.argmax(fft_vals)
            ]

            X.append(features)
            y.append(label_clean)
            
# FINAL DATA CHECK
X = np.array(X)
y = np.array(y)

print("\nDATA SUMMARY")
print("Samples:", len(X))
print("Classes:", Counter(y))

if len(set(y)) < 2:
    print(" ERROR: Only one class present")
    exit()


# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# FEATURE SCALING 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# MODELS

models = {
    "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}


# TRAIN + EVALUATE

for name, model in models.items():
    print(f"\n==== {name} ====")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 (weighted):", f1_score(y_test, y_pred, average="weighted"))
    print("F1 (macro):", f1_score(y_test, y_pred, average="macro"))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    
    # ROC CURVE

    classes = np.unique(y)
    y_test_bin = label_binarize(y_test, classes=classes)

    y_score = model.predict_proba(X_test)

    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, label=classes[i])

    plt.title(f"ROC - {name}")
    plt.legend()
    plt.show()

# REAL SIGNAL VISUALIZATION (FIXED)

sample_signal = raw_windows[0]

plt.plot(sample_signal)
plt.title("Real Signal Window")
plt.show()

plt.hist(sample_signal, bins=30)
plt.title("Signal Distribution")
plt.show()

# CLUSTERING
kmeans = KMeans(n_clusters=len(set(y)), random_state=42)
clusters = kmeans.fit_predict(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
plt.title("Clustering (PCA)")
plt.show()