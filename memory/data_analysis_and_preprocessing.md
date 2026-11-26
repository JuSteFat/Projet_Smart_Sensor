# Mémoire projet — Analyse des données & Prétraitement (Smart Sensor)

## 1. Contexte matériel & contraintes cibles
- **MCU** : STM32N657X0 (Arm Cortex-M55 @ 800 MHz, 4.2 Mo SRAM, NPU Neural-ART 600 GOPS).
- **Capteur** : Accéléromètre NXP FXLS8974CF (données enregistrées sur 2 axes `X, Y` dans la base S8).
- **Cible de déploiement** : Modèle **TFLite quantifié INT8** (opérations supportées par le NPU : Conv1D/2D, Depthwise/Separable Conv, Dense, Pooling, ReLU/Softmax ; éviter RNN/LSTM/Transformers).
- **Règle d'or** : Tout prétraitement appliqué à l'entraînement en Python doit rester simple et reproductible en C sur le Cortex-M55 avant l'inférence NPU.

---

## 2. Diagnostic des données (`Pro_S9_Smart_Sensor_Project/`)

### Ce qui est sain
- **2563 fichiers CSV** au total (`254` à `259` par classe `0..9`), tous valides (0 `NaN`/`Inf`, 0 fichier corrompu).
- **2 colonnes entières** : `X,Y` (pas 3).
- **0 doublon exact** (MD5), ni intra-classe ni inter-classes.

### Les 5 problèmes d'hétérogénéité (acquisitions multi-élèves)
1. **Zero-padding massif en fin de fichier (~32 % du dataset / 815 fichiers)** :
   - Lot `T = 3286` (`710` fichiers, `71`/classe, ex. `*_1.csv`) : signal utile médian de **1143 pts**, suivi d'environ **2143 zéros `(0, 0)`** (~65 % de vide).
   - Lot `T = 1600` (`105` fichiers, ex. `*(3).csv`) : signal utile médian de **74 pts** (`39` à `99`), suivi d'environ **1526 zéros `(0, 0)`** (~95 % de vide).
2. **Fréquences d'échantillonnage et bruit haute fréquence hétérogènes** :
   - Lot `*(2).csv` (`695` fichiers) : très lisse (`std(diff)/std(x) ≈ 0.25`, ~17 % d'énergie HF).
   - Lot `*_1.csv` (`710` fichiers) : bruité (`std(diff)/std(x) ≈ 0.94`, ~34 % d'énergie HF, frottements souris/bureau).
   - Lot `T = 1600` (`105` fichiers) : très peu de points (~74 pts) et très bruité (`≈ 1.10`, ~73 % d'énergie HF).
3. **Écarts d'amplitude (10× à 40×) et d'offset DC entre fichiers** :
   - `core_std_x` varie de `8.6` à `727` (`p10 = 56`, `p90 = 310`), `core_max` jusqu'à `2039`.
   - `core_mean_x` dans `[-106, +50]` et `core_mean_y` dans `[-44, +68]`.
4. **Temps morts (silences non nuls) en début/fin** :
   - Sur les 1748 fichiers non paddés, les 10 % les plus extrêmes contiennent **12 % à 52 %** de bruit de repos au début et jusqu'à **43 %** à la fin.
5. **Perte de l'ordre temporel avec `GlobalAveragePooling`** :
   - Des chiffres comme `6`/`9` ou `2`/`5` partagent des motifs locaux similaires mais dans un ordre chronologique différent.

---

## 3. Choix de prétraitement et justifications

| Étape | Pourquoi côté ML / Signal | Pourquoi côté STM32N6 |
|---|---|---|
| **1. Suppression du zero-padding (`strip_zero_padding`)** | Évite qu'un chiffre occupe 100 %, 35 % ou 5 % de la fenêtre selon le lot d'élève. Aligne l'échelle temporelle pour les filtres CNN. | Sur la carte, la capture démarre/s'arrête sur le geste : il n'y a pas de padding artificiel. |
| **2. Trimming des temps morts (`trim_quiet_margins`)** | Aligne le début et la fin du tenseur sur le vrai mouvement, réduisant la variance inutile sur ~200 exemples/classe. | Simple seuillage sur l'amplitude centrée, trivial en C. |
| **3. Filtrage passe-bas adaptatif (`lowpass_moving_average`)** | Supprime le bruit de frottement HF, homogénéise les lots lisses/bruités et évite l'aliasing avant sous-échantillonnage vers `SEQ_LEN = 128`. | Moyenne glissante très peu coûteuse sur Cortex-M55. |
| **4. Normalisation par échantillon à norme commune (`normalize_sample`)** | Retire l'offset DC de chaque capture et normalise `X` et `Y` par une **norme RMS commune** $\sqrt{\text{mean}(x^2 + y^2)}$ : invariant à la force du geste tout en préservant le ratio horizontal/vertical (ex. ne pas amplifier le bruit `X` sur un chiffre `1`). Améliore aussi la dynamique en INT8. | 2 moyennes + 1 norme RMS sur `128` points = quelques µs. |
| **5. `Flatten` après convolutions (`SEQ_LEN = 128`, 2–3 blocs, `kernel_size` 3 ou 5)** | Préserve l'ordre chronologique (début / milieu / fin du chiffre) tout en gardant un vecteur plat compact (quelques centaines de valeurs) avant `Dense(10)`. | Taille modèle TFLite INT8 de quelques dizaines de Ko (très loin des 4.2 Mo SRAM). |

---

## 4. Benchmarks Séance 1 (même prétraitement `SEQ_LEN = 128`, même split 80/20 stratifié)

| Modèle | Script | Val acc | Test FP32 | Test TFLite INT8 | Paramètres | Taille INT8 |
|---|---|---:|---:|---:|---:|---:|
| **CNN 1D** (2 blocs, k=3, 64→32, Flatten) | `poc_autokeras.py` | 93.17 % | **95.13 %** | **95.32 %** | **16 234** | **25.6 Ko** |
| **MLP** (2 couches, 128→32) | `poc_mlp.py` | 95.12 % | 94.74 % | 94.74 % | 37 354 | 44.4 Ko |
| **Hodographe CNN 2D** (`32×32×2`, 3 blocs 3×3, 64→32→32) | `poc_hodographe.py` | 94.15 % | 93.76 % | 93.76 % | 30 218 | 38.3 Ko |
| **CNN 1D 4 canaux** (`ax, ay, vx, vy`, 2 blocs, k=3, lr=1e-4) | `poc_cnn1d_acc_vel.py` | 90.73 % | 93.18 % | 92.79 % | 16 618 | 25.9 Ko |

**Conclusion** : Le **CNN 1D** sur les séries temporelles prétraitées `(128, 2)` reste le grand gagnant sur tous les critères : meilleure accuracy test INT8 (**95.32 %**), **~2× moins de paramètres**, **le plus léger (25.6 Ko)** et prétraitement plus simple en C sur STM32 (pas d'intégration ni de rasterisation 2D). L'hodographe 2D (93.76 %) confirme que l'information de vitesse est pertinente, mais la projection sur grille `32×32` perd en résolution temporelle (auto-croisements qui écrasent le canal `t/T`) et reste sensible au biais résiduel d'intégration.

---

## 5. Notes techniques (environnement)
- **Python 3.11 + TensorFlow 2.21 + Keras 3.15 + AutoKeras 3.0.0** dans `venv/`.
- **Bug connu contourné** : dans `poc_autokeras.py`, `SafeClassificationHead` caste `self.shape` en `int` Python natif car Keras 3.15 rejette `np.int64` dans `layers.Dense(units)`.
- **Dépendance** : `tensorboard` doit être présent dans `requirements.txt` (requis par `keras_tuner` avec TF 2.21).
