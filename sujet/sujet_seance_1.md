# TP – Smart Sensor

3A IATI-TSE
Julien Marc, Blaise MULLIEZ

## Introduction

Le projet Smart Sensor consiste à développer une chaîne d'instrumentation intelligente, de l'acquisition des données au déploiement d'un modèle sur MCU, en passant par l'entraînement dudit modèle et son optimisation pour l'inférence embarquée.

L'application développée pendant ce projet est une application embarquée de classification de chiffres tracés à l'aide d'une souris, exploitant des données issus d'un accéléromètre placé dans cette souris.

Le matériel dont vous disposez est :

— Une carte de développement STM32 NUCLEO-144 N657X0-Q, embarquant notamment un MCU STM32N657X0 regroupant entre autres un coeur haute performance Arm Cortex-M55 MCU cadencé à 800MHz, 4.2MB SRAM, et un accélérateur Neural-ART à 600 GOPS ;
— Un accéléromètre compact 3 axes NXP FXLS8974CF, avec un mode haute performance et haute précision.

![img-0.jpeg](img-0.jpeg)

FIGURE 1 – Carte de développement STM32 NUCLEO-144 N657X0-Q

Les détails de la carte STM32 NUCLEO-144 N657X0-Q sont consultables ici.

Les détails de l'accélérateur Neural-ART sont consultables ici.

Vous disposez en outre de Python, VSCode et de la suite logiciel STM32Cube pour entraîner votre modèle, le convertir dans le format adéquat, et le programmer sur le MCU.

1

# 1 Développement d'un modèle de classification

Dans cette première étape du projet, vous devrez entraîner un modèle en prenant en compte les contraintes du composant sur lequel il sera déployé.

## 1.1 Récupération de la base de données

Vous avez créé en S8 une base de données de chiffres tracés à l'aide de la souris. Toutes les captures ont été regroupées pour alimenter votre base de données d'entraînement.

Vous pouvez récupérer les données sur Github via la commande (exécutée dans un terminal) :

$ git clone https://github.com/blaisemulliez/Pro_S9_Smart_Sensor_Project.git
$ cd Pro_S9_Smart_Sensor_Project

Remarque : si git n'est pas installé, installez-le avec les commandes :

$ sudo apt update
$ sudo apt install git

La base de données contient 2575 échantillons, soit un peu plus de 250 par classe.

## 1.2 Contraintes liées au NPU

A présent, vous devez entraîner un modèle à classifier ces données, en prenant en compte les contraintes du MCU STM32N657X0, et notamment des limites de l'accélérateur Neural-ART. En particulier

### 1.2.1 Fonctions réalisables dans le NPU

Le NPU (Neural Processing Unit) de la série STM32N6 est conçu pour accélérer l'inférence de réseaux de neurones. Il prend en charge les opérations courantes suivantes :

- Convolutions (2D, depthwise, pointwise, etc.)
- Fully connected (couches denses)
- Pooling (max, average, global)
- Activation functions (ReLU, sigmoid, tanh, softmax, etc.)
- Normalization (Batch Normalization, Layer Normalization)
- Element-wise operations (addition, multiplication, concatenation)
- Element-wise operations (addition, multiplication, concatenation)
- Reshape / transpose / slicing
- Quantization / dequantization (entre entiers et flottants)

### 1.2.2 Types de quantification supportés

Le NPU Neural-ART fonctionne principalement sur des données entières pour maximiser les performances. Les formats pris en charge incluent :

- INT8 (symétrique et asymétrique) – le plus courant
- UINT8
- INT16
- INT32 (pour accumulations intermédiaires)

### 1.2.3 Limitations

Le NPU n'est pas adapté aux opérations de type récurrent (RNN, LSTM) complexes ou aux architectures très irrégulières (certains transformers). Les fonctions d'activation exotiques (gelu, swish) peuvent nécessiter une approximation en INT8 ou être exécutées sur le processeur ARM principal (Cortex-M).

2

### 1.2.4 Développement du modèle

En prenant en compte les considérations ci-dessus, développez un modèle pour réaliser la classification des données récupérées sur GitHub.

Ce modèle sera ensuite converti dans le format TensorFlowLite dédié aux systèmes embarqués. Les outils de conversion et d'optimisation fournis par ST vous permettent de développer votre modèle dans le format et avec le framework qui vous convient : Keras, TensorFlow, PyTorch, Sci-Kit Learn.

FIN DE LA SEANCE 1

3

