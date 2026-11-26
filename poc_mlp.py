from pathlib import Path
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


def main() -> None:
    # 1. Même chargement, prétraitement et split que poc_autokeras.py (comparaison équitable)
    print(f"Chargement et prétraitement du dataset depuis {DATA_DIR} (SEQ_LEN={SEQ_LEN})...")
    X, y = load_dataset(DATA_DIR, SEQ_LEN)
    print(
        f"Dataset prêt : X={X.shape}, y={y.shape}, "
        f"min={X.min():.2f}, max={X.max():.2f}, std={X.std():.2f}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 2. Espace de recherche MLP pur (100 % Dense + BatchNorm + ReLU, NPU-friendly)
    # DenseBlock aplatit automatiquement (128, 2) -> 256 entrées
    input_node = ak.Input()
    x = ak.DenseBlock(
        num_layers=hp.Choice("num_layers", [1, 2, 3], default=2),
        num_units=hp.Choice("num_units", [32, 64, 128, 256], default=128),
        use_batchnorm=None,  # Laisse AutoKeras tester avec / sans BatchNorm
        dropout=hp.Choice("dropout", [0.0, 0.2, 0.4], default=0.2),
    )(input_node)
    output_node = SafeClassificationHead(num_classes=NUM_CLASSES)(x)

    # 3. Recherche d'architecture (projet séparé pour ne pas écraser auto_model du CNN)
    clf = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        project_name="auto_model_mlp",
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

    # 4. Évaluation et export du meilleur modèle MLP
    loss, acc = clf.evaluate(X_test, y_test)
    best_model = clf.export_model()
    best_model.summary()

    keras_path = Path("best_mlp_model.keras")
    best_model.save(keras_path)

    # 5. Export et validation TFLite INT8 (cible NPU Neural-ART)
    tflite_path = Path("best_mlp_model_int8.tflite")
    export_tflite_int8(best_model, X_train, tflite_path)
    tflite_acc = evaluate_tflite_int8(tflite_path, X_test, y_test)
    tflite_kb = tflite_path.stat().st_size / 1024

    # 6. Récapitulatif bien visible du meilleur modèle MLP retenu
    best_trial = clf.tuner.oracle.get_best_trials(num_trials=1)[0]
    best_hp = best_trial.hyperparameters.values
    print("\n" + "=" * 70)
    print("  MEILLEUR MODÈLE MLP RETENU PAR AUTOKERAS")
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
