from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
import autokeras as ak
from keras_tuner.engine import hyperparameters as hp
from sklearn.model_selection import train_test_split


@keras.utils.register_keras_serializable(package="poc_autokeras")
class SafeClassificationHead(ak.ClassificationHead):
    """Contourne un bug AutoKeras 3 + Keras 3.15 où self.shape[-1] est un np.int64."""

    def build(self, hp, inputs=None):
        if self.shape is not None:
            self.shape = tuple(int(d) if d is not None else None for d in self.shape)
        return super().build(hp, inputs)


# --- Configuration ---
DATA_DIR = Path("Pro_S9_Smart_Sensor_Project")
NUM_CLASSES = 10
SEQ_LEN = 128          # Puissance de 2 : léger sur MCU, conserve l'ordre temporel après pooling
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_TRIALS = 15
EPOCHS = 40
BATCH_SIZE = 32


def strip_zero_padding(seq: np.ndarray) -> np.ndarray:
    """Supprime les lignes (0, 0) de padding en début et fin de fichier."""
    nonzero = np.where((seq[:, 0] != 0.0) | (seq[:, 1] != 0.0))[0]
    if len(nonzero) == 0:
        return seq
    return seq[nonzero[0] : nonzero[-1] + 1]


def trim_quiet_margins(
    seq: np.ndarray, threshold_ratio: float = 0.12, margin_frac: float = 0.02
) -> np.ndarray:
    """Recadre la séquence sur la zone active (supprime les temps morts début/fin)."""
    if len(seq) < 16:
        return seq

    centered = seq - np.median(seq, axis=0)
    mag = np.linalg.norm(centered, axis=1)
    p95 = float(np.percentile(mag, 95))
    if p95 <= 1e-6:
        return seq

    active = np.where(mag > threshold_ratio * p95)[0]
    if len(active) < 8:
        return seq

    margin = max(1, int(round(margin_frac * len(seq))))
    start = max(0, int(active[0]) - margin)
    end = min(len(seq), int(active[-1]) + 1 + margin)
    return seq[start:end]


def lowpass_moving_average(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Filtre passe-bas (moyenne glissante) anti-aliasing et réduction du bruit HF."""
    # Fenêtre adaptée au facteur de sous-échantillonnage, avec un minimum de 3 points
    win = max(3, int(round(len(seq) / target_len)))
    if win % 2 == 0:
        win += 1
    if len(seq) <= win:
        return seq

    kernel = np.ones(win, dtype=np.float32) / win
    pad = win // 2
    out = np.empty_like(seq, dtype=np.float32)
    for c in range(seq.shape[1]):
        padded = np.pad(seq[:, c], (pad, pad), mode="edge")
        out[:, c] = np.convolve(padded, kernel, mode="valid")
    return out


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Rééchantillonne une séquence (T, C) vers (target_len, C) par interpolation linéaire."""
    orig_len, num_channels = seq.shape
    if orig_len == target_len:
        return seq.astype(np.float32)

    orig_t = np.linspace(0.0, 1.0, orig_len, dtype=np.float32)
    target_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    resampled = np.empty((target_len, num_channels), dtype=np.float32)
    for c in range(num_channels):
        resampled[:, c] = np.interp(target_t, orig_t, seq[:, c])
    return resampled


def normalize_sample(seq: np.ndarray) -> np.ndarray:
    """Centre chaque axe (retire l'offset DC) et normalise par une norme RMS commune X/Y."""
    centered = seq - seq.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))) + 1e-8
    return (centered / rms).astype(np.float32)


def preprocess_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Pipeline complet de nettoyage et normalisation (reproductible en C sur STM32)."""
    seq = strip_zero_padding(seq)
    seq = trim_quiet_margins(seq)
    seq = lowpass_moving_average(seq, target_len)
    seq = resample_sequence(seq, target_len)
    seq = normalize_sample(seq)
    return seq


def load_dataset(data_dir: Path, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Charge et prétraite tous les CSV (colonnes X, Y) depuis les sous-dossiers 0..9."""
    samples = []
    labels = []

    for label in range(NUM_CLASSES):
        class_dir = data_dir / str(label)
        csv_files = sorted(class_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {class_dir}")

        for csv_path in csv_files:
            seq = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
            if seq.ndim != 2 or seq.shape[1] != 2:
                raise ValueError(f"Format inattendu dans {csv_path}: shape={seq.shape}")
            samples.append(preprocess_sequence(seq, seq_len))
            labels.append(label)

    X = np.stack(samples, axis=0)            # (N, SEQ_LEN, 2) en float32
    y = np.array(labels, dtype=np.int32)     # (N,) avec labels 0..9
    return X, y


def export_tflite_int8(
    model: tf.keras.Model,
    X_calib: np.ndarray,
    output_path: Path,
    num_calib_samples: int = 200,
) -> None:
    """Convertit le modèle Keras en TFLite quantifié INT8 (compatible NPU Neural-ART)."""
    def representative_dataset():
        for i in range(min(num_calib_samples, len(X_calib))):
            yield [X_calib[i : i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    print(f"Modèle TFLite INT8 sauvegardé : {output_path} ({len(tflite_model) / 1024:.1f} Ko)")


def evaluate_tflite_int8(tflite_path: Path, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Évalue l'accuracy du modèle TFLite INT8 sur le jeu de test."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_scale, in_zero = input_details["quantization"]

    correct = 0
    for i in range(len(X_test)):
        sample = X_test[i : i + 1]
        q_sample = np.clip(np.round(sample / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details["index"], q_sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        if int(np.argmax(output)) == int(y_test[i]):
            correct += 1

    acc = correct / len(X_test)
    print(f"TFLite INT8 test accuracy: {acc:.4f}")
    return acc


def main() -> None:
    # 1. Chargement et prétraitement des données (2563 échantillons, 2 axes X/Y)
    print(f"Chargement et prétraitement du dataset depuis {DATA_DIR} (SEQ_LEN={SEQ_LEN})...")
    X, y = load_dataset(DATA_DIR, SEQ_LEN)
    print(
        f"Dataset prêt : X={X.shape}, y={y.shape}, "
        f"min={X.min():.2f}, max={X.max():.2f}, std={X.std():.2f}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 2. Espace de recherche contraint (NPU-friendly, léger, préserve l'ordre temporel)
    input_node = ak.Input()
    x = ak.ConvBlock(
        kernel_size=hp.Choice("kernel_size", [3, 5], default=3),
        num_blocks=hp.Choice("num_blocks", [2, 3], default=3),
        num_layers=hp.Choice("num_layers", [1, 2], default=1),
        filters=hp.Choice("filters", [16, 32, 64], default=32),
        max_pooling=True,
        separable=None,   # Laisse AutoKeras tester Conv standard et Depthwise Separable
        dropout=0.2,
    )(input_node)
    # Flatten (et non GlobalAveragePooling) pour conserver l'ordre chronologique du tracé
    x = ak.Flatten()(x)
    output_node = SafeClassificationHead(num_classes=NUM_CLASSES)(x)

    # 3. Recherche d'architecture
    clf = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        overwrite=True,
        max_trials=MAX_TRIALS,
        objective="val_accuracy",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        )
    ]

    clf.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # 4. Évaluation et export du meilleur modèle Keras
    loss, acc = clf.evaluate(X_test, y_test)
    best_model = clf.export_model()
    best_model.summary()

    keras_path = Path("best_autokeras_model.keras")
    best_model.save(keras_path)

    # 5. Export et validation TFLite INT8 (cible NPU Neural-ART)
    tflite_path = Path("best_autokeras_model_int8.tflite")
    export_tflite_int8(best_model, X_train, tflite_path)
    tflite_acc = evaluate_tflite_int8(tflite_path, X_test, y_test)
    tflite_kb = tflite_path.stat().st_size / 1024

    # 6. Récapitulatif bien visible du meilleur modèle retenu
    best_trial = clf.tuner.oracle.get_best_trials(num_trials=1)[0]
    best_hp = best_trial.hyperparameters.values
    print("\n" + "=" * 70)
    print("  MEILLEUR MODÈLE RETENU PAR AUTOKERAS")
    print("=" * 70)
    print(f"  Trial ID          : {best_trial.trial_id}")
    print(f"  Best val_accuracy : {best_trial.score:.4f}")
    print(f"  Test FP32         : accuracy={acc:.4f} (loss={loss:.4f})")
    print(f"  Test TFLite INT8  : accuracy={tflite_acc:.4f} (écart={tflite_acc - acc:+.4f})")
    print(f"  Paramètres        : {best_model.count_params():,}")
    print(f"  Taille TFLite INT8: {tflite_kb:.1f} Ko")
    print(f"  Fichier Keras     : {keras_path.resolve()}")
    print(f"  Fichier TFLite    : {tflite_path.resolve()}")
    print("  Hyperparamètres   :")
    for k, v in best_hp.items():
        print(f"    - {k}: {v}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
