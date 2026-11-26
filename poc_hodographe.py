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

IMG_SIZE = 32  # Image 32x32x2 (trace de l'hodographe + progression temporelle)


def accel_to_hodograph_image(acc: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Construit l'image 2D de l'hodographe des vitesses (vx, vy) à partir de
    l'accélération prétraitée (SEQ_LEN, 2).

    - Intégration simple : v(t) = cumsum(acc). Comme acc est déjà centrée (moyenne=0),
      la dérive linéaire est annulée (v(0) ≈ v(T) ≈ 0).
    - Canal 0 : trace binaire/continue de la courbe (vx, vy) dans le plan 2D.
    - Canal 1 : progression temporelle t/T dans [0, 1] pour conserver le sens de parcours.
    """
    vel = np.cumsum(acc, axis=0)  # (T, 2)

    # Centrage de la boîte englobante avec échelle commune (préserve le ratio vx/vy)
    vmin = vel.min(axis=0)
    vmax = vel.max(axis=0)
    center = 0.5 * (vmin + vmax)
    vel_centered = vel - center
    half_span = float(np.max(np.abs(vel_centered))) + 1e-8

    # Projection dans [1, img_size - 2] (marge de 1 pixel)
    coords = (vel_centered / half_span) * ((img_size - 3) / 2.0) + ((img_size - 1) / 2.0)

    img = np.zeros((img_size, img_size, 2), dtype=np.float32)
    T = len(coords)

    # Tracé continu entre points consécutifs (évite les trous sur les accélérations rapides)
    for t in range(T - 1):
        p0 = coords[t]
        p1 = coords[t + 1]
        dist = float(np.max(np.abs(p1 - p0)))
        steps = max(1, int(np.ceil(dist * 2)))
        t_norm = t / max(1, T - 1)
        for s in range(steps + 1):
            alpha = s / steps
            px = int(np.clip(round(p0[0] + alpha * (p1[0] - p0[0])), 0, img_size - 1))
            py = int(np.clip(round(p0[1] + alpha * (p1[1] - p0[1])), 0, img_size - 1))
            img[py, px, 0] = 1.0
            img[py, px, 1] = t_norm

    return img


def build_hodograph_dataset(X_acc: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """Convertit un dataset d'accélérations (N, T, 2) en images hodographes (N, H, W, 2)."""
    return np.stack([accel_to_hodograph_image(seq, img_size) for seq in X_acc], axis=0)


def main() -> None:
    # 1. Chargement et prétraitement identiques à poc_autokeras.py
    print(f"Chargement du dataset depuis {DATA_DIR} (SEQ_LEN={SEQ_LEN})...")
    X_acc, y = load_dataset(DATA_DIR, SEQ_LEN)

    print(f"Génération des hodographes des vitesses ({IMG_SIZE}x{IMG_SIZE}x2)...")
    X_img = build_hodograph_dataset(X_acc, IMG_SIZE)
    print(f"Dataset hodographe prêt : X={X_img.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_img, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 2. Espace de recherche CNN 2D (100 % natif NPU Neural-ART)
    input_node = ak.Input()
    x = ak.ConvBlock(
        kernel_size=3,  # Noyau 3x3 standard pour images 32x32 (32 -> 16 -> 8 -> 4)
        num_blocks=hp.Choice("num_blocks", [2, 3], default=3),
        num_layers=hp.Choice("num_layers", [1, 2], default=1),
        filters=hp.Choice("filters", [16, 32, 64], default=32),
        max_pooling=True,
        separable=None,  # Teste Conv2D standard et Depthwise Separable Conv2D
        dropout=0.2,
    )(input_node)
    x = ak.Flatten()(x)
    output_node = SafeClassificationHead(num_classes=NUM_CLASSES)(x)

    # 3. Recherche d'architecture (projet séparé)
    clf = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        project_name="auto_model_hodographe",
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

    keras_path = Path("best_hodographe_model.keras")
    best_model.save(keras_path)

    # 5. Export et validation TFLite INT8 (cible NPU Neural-ART)
    tflite_path = Path("best_hodographe_model_int8.tflite")
    export_tflite_int8(best_model, X_train, tflite_path)
    tflite_acc = evaluate_tflite_int8(tflite_path, X_test, y_test)
    tflite_kb = tflite_path.stat().st_size / 1024

    # 6. Récapitulatif bien visible du meilleur modèle retenu
    best_trial = clf.tuner.oracle.get_best_trials(num_trials=1)[0]
    best_hp = best_trial.hyperparameters.values
    print("\n" + "=" * 70)
    print("  MEILLEUR MODÈLE HODOGRAPHE (CNN 2D) RETENU PAR AUTOKERAS")
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
