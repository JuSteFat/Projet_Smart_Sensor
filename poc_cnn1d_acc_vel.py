from pathlib import Path
import numpy as np
import tensorflow as tf
import autokeras as ak
from keras_tuner.engine import hyperparameters as hp
from sklearn.model_selection import train_test_split

from poc_autokeras import (
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS,
    MAX_TRIALS,
    NUM_CLASSES,
    RANDOM_STATE,
    SEQ_LEN,
    TEST_SIZE,
    SafeClassificationHead,
    evaluate_tflite_int8,
    export_tflite_int8,
    load_dataset,
)


def add_velocity_channels(X_acc: np.ndarray) -> np.ndarray:
    """
    Ajoute les vitesses intégrées (vx, vy) aux accélérations (ax, ay) -> (N, T, 4).

    Comme (ax, ay) est déjà centré (moyenne=0), l'intégration cumsum annule la dérive
    linéaire (v(0) ≈ v(T) ≈ 0). On centre ensuite (vx, vy) et on normalise par une norme
    RMS commune X/Y pour que les 4 canaux aient la même dynamique (idéal pour INT8).
    """
    vel = np.cumsum(X_acc, axis=1)  # (N, T, 2)
    vel = vel - vel.mean(axis=1, keepdims=True)
    rms_v = np.sqrt(np.mean(np.sum(vel**2, axis=2, keepdims=True), axis=1, keepdims=True)) + 1e-8
    vel = (vel / rms_v).astype(np.float32)
    return np.concatenate([X_acc, vel], axis=-1).astype(np.float32)


def main() -> None:
    # 1. Chargement, prétraitement et ajout des canaux vitesse (ax, ay, vx, vy)
    print(f"Chargement du dataset depuis {DATA_DIR} (SEQ_LEN={SEQ_LEN})...")
    X_acc, y = load_dataset(DATA_DIR, SEQ_LEN)
    X = add_velocity_channels(X_acc)
    print(
        f"Dataset 4 canaux prêt : X={X.shape}, y={y.shape}, "
        f"min={X.min():.2f}, max={X.max():.2f}, std={X.std():.2f}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 2. Même espace de recherche CNN 1D que poc_autokeras.py (comparaison équitable)
    input_node = ak.Input()
    x = ak.ConvBlock(
        kernel_size=hp.Choice("kernel_size", [3, 5], default=3),
        num_blocks=hp.Choice("num_blocks", [2, 3], default=3),
        num_layers=hp.Choice("num_layers", [1, 2], default=1),
        filters=hp.Choice("filters", [16, 32, 64], default=32),
        max_pooling=True,
        separable=None,   # Teste Conv1D standard et Depthwise Separable
        dropout=0.2,
    )(input_node)
    x = ak.Flatten()(x)
    output_node = SafeClassificationHead(num_classes=NUM_CLASSES)(x)

    # 3. Recherche d'architecture (projet séparé)
    clf = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        project_name="auto_model_cnn1d_4ch",
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

    keras_path = Path("best_cnn1d_4ch_model.keras")
    best_model.save(keras_path)

    # 5. Export et validation TFLite INT8 (cible NPU Neural-ART)
    tflite_path = Path("best_cnn1d_4ch_model_int8.tflite")
    export_tflite_int8(best_model, X_train, tflite_path)
    tflite_acc = evaluate_tflite_int8(tflite_path, X_test, y_test)
    tflite_kb = tflite_path.stat().st_size / 1024

    # 6. Récapitulatif bien visible du meilleur modèle retenu
    best_trial = clf.tuner.oracle.get_best_trials(num_trials=1)[0]
    best_hp = best_trial.hyperparameters.values
    print("\n" + "=" * 70)
    print("  MEILLEUR MODÈLE CNN 1D 4 CANAUX (ax, ay, vx, vy)")
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
