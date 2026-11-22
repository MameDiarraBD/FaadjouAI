# Faadjou — document technique

**Projet :** reconnaissance vocale Wolof pour la plateforme de prise de rendez-vous Faadjou  
**Concours :** PAS Challenge 2026  
**Objet :** données, méthode d'entraînement, résultats ASR et déploiement  
**Date des mesures :** 22 novembre 2026

## 1. Résumé exécutif

L'entraînement Kaggle v2 a produit un modèle Whisper Wolof ajusté sur 10 000
exemples d'entraînement. Le modèle a été évalué sur 300 exemples de test
séparés du jeu d'entraînement.

Le modèle de base `M9and2M/whisper-small-wolof` obtient :

- WER : **0,7346**
- CER : **0,4767**
- Temps moyen : **0,419 seconde/exemple**

Le modèle ajusté `whisper-wolof-ft` obtient :

- WER : **0,5194**
- CER : **0,2935**
- Temps moyen : **0,180 seconde/exemple**

L'amélioration absolue du WER est de **0,2152**, soit une baisse mesurée de
**29,3 %** par rapport au WER du modèle de base sur le même jeu de test.

La décision enregistrée dans Kaggle est : **CONSERVER le modèle ajusté**.
Le meilleur point de validation a été atteint au pas d'entraînement 1500,
avec un WER de validation de **0,4134295227524972**.

Ce gain confirme une amélioration sur le jeu de test mesuré. Il ne démontre
pas à lui seul une capacité clinique suffisante : le modèle reste soumis à
des tests supplémentaires sur des phrases médicales, des locuteurs, des
accents et des conditions acoustiques réelles.

## 2. Contexte et objectif

Faadjou Wax vise à permettre à un patient d'obtenir une orientation médicale
après une prise de parole en Wolof. Cette fonctionnalité comporte plusieurs
étapes :

1. Capture du signal vocal.
2. Reconnaissance automatique de la parole Wolof.
3. Traduction Wolof vers français.
4. Orientation structurée vers une spécialité.
5. Application séparée des règles de signaux d'alarme.
6. Proposition vers une consultation fictive ou réelle selon l'intégration.

Le modèle entraîné couvre uniquement la reconnaissance vocale. Il ne fait pas
de diagnostic et ne remplace pas un professionnel de santé.

## 3. Données

### 3.1 Données utilisées

La préparation finale a utilisé le corpus Hugging Face
`vonewman/wolof-audio-data`, identifié dans le journal Kaggle.

Volumes rapportés :

| Élément | Nombre |
|---|---:|
| Exemples disponibles avant retrait des recoupements | 12 300 |
| Exemples d'entraînement retenus | 10 000 |
| Exemples de validation | 300 |
| Exemples de test Hugging Face | 300 |
| Recoupements retirés vis-à-vis du test | 297 |
| Recoupements retirés vis-à-vis de la validation | 309 |

Le résumé machine de Kaggle est disponible dans
`results/data_summary.json`.

### 3.2 Séparation des jeux

Les données ont été séparées avant l'entraînement :

- un jeu d'entraînement de 10 000 exemples ;
- un jeu de validation de 300 exemples ;
- un jeu de test de 300 exemples ;
- des contrôles anti-fuite supprimant les transcriptions en recoupement avec
  les jeux de test ou de validation.

La validation n'a pas été utilisée pour apprendre les poids du modèle.
Elle a été utilisée pour mesurer le WER périodiquement et conserver le
meilleur checkpoint.

### 3.3 Protection contre la fuite de données

Le journal Kaggle indique que :

- 297 exemples ont été retirés lors du contrôle vis-à-vis du test ;
- 309 exemples ont été retirés lors du contrôle vis-à-vis de la validation ;
- les recoupements ont été comparés après normalisation des transcriptions.

Cette mesure réduit le risque que les scores soient gonflés par des phrases
présentes à la fois dans l'entraînement et l'évaluation.

### 3.4 Limites des données

Les données disponibles ne permettent pas d'affirmer :

- une couverture équilibrée de tous les dialectes Wolof ;
- une robustesse validée à tous les accents ;
- une représentation suffisante de l'environnement clinique ;
- une validation sur un jeu médical indépendant ;
- une garantie de compréhension des termes médicaux rares.

Les résultats doivent donc être présentés comme une mesure technique sur les
300 exemples du test Hugging Face, pas comme une performance clinique
généralisée.

## 4. Méthode d'entraînement

### 4.1 Modèle de base

Le modèle de départ est :

`M9and2M/whisper-small-wolof`

Il s'agit d'un modèle Whisper adapté au Wolof. Son évaluation initiale fournit
la référence servant à mesurer le gain du fine-tuning.

### 4.2 Environnement

D'après le journal d'exécution :

- plateforme : Kaggle ;
- GPU : Tesla T4 ;
- PyTorch : `2.11.0+cu130` ;
- Transformers : `5.20.2` ;
- entraînement lancé le 22 novembre 2026.

### 4.3 Configuration

La configuration finale observée utilise :

- 10 000 exemples d'entraînement ;
- 300 exemples de validation ;
- maximum 3 passages sur les données ;
- évaluation toutes les 250 étapes ;
- taux d'apprentissage : `1e-5` ;
- patience : 3 évaluations ;
- budget maximal annoncé : 100 minutes ;
- durée réelle de l'entraînement : 5 369 secondes, soit environ 89,5 minutes ;
- 1 875 étapes maximales ;
- meilleure sauvegarde conservée au checkpoint 1500.

### 4.4 Protocole

Le protocole comprend :

1. téléchargement du jeu de test Hugging Face ;
2. téléchargement du jeu d'entraînement Hugging Face ;
3. retrait des recoupements ;
4. séparation entre entraînement, validation et test ;
5. test rapide de charge, préparation des données et entraînement ;
6. évaluation du modèle de base ;
7. fine-tuning sur le jeu d'entraînement ;
8. évaluation périodique sur la validation ;
9. conservation du meilleur checkpoint selon le WER de validation ;
10. évaluation finale du modèle ajusté sur le test ;
11. comparaison avec le modèle de base ;
12. décision de conservation ou de rejet.

### 4.5 Mesures utilisées

Deux métriques principales ont été enregistrées :

- **WER** : taux d'erreur sur les mots ;
- **CER** : taux d'erreur sur les caractères.

Les prédictions brutes, les références et les scores sont conservés dans les
fichiers CSV Kaggle.

## 5. Résultats

### 5.1 Comparaison des modèles

| Modèle | Jeu | Exemples | WER | CER | Temps/exemple |
|---|---|---:|---:|---:|---:|
| M9and2M/whisper-small-wolof | test_hf | 300 | 0,7346 | 0,4767 | 0,419 s |
| whisper-wolof-ft | test_hf | 300 | 0,5194 | 0,2935 | 0,180 s |

La comparaison exacte est dans `results/asr_finetune_compare.csv`.

### 5.2 Gain obtenu

- gain absolu WER : `0,7346 - 0,5194 = 0,2152` ;
- baisse relative du WER : environ `29,3 %` ;
- gain absolu CER : `0,4767 - 0,2935 = 0,1832` ;
- temps moyen divisé d'environ `2,33` par rapport au modèle de base.

### 5.3 Validation pendant l'entraînement

Quelques points de validation mesurés :

| Étape | WER | CER |
|---:|---:|---:|
| 250 | 0,4931 | 0,2487 |
| 500 | 0,5158 | 0,2694 |
| 750 | 0,4553 | 0,2442 |
| 1000 | 0,4811 | 0,2869 |
| 1250 | 0,4351 | 0,2444 |
| 1500 | 0,4134 | 0,2020 |
| 1750 | 0,4473 | 0,2517 |
| 1875 | 0,4448 | 0,2494 |

Le meilleur point est donc obtenu au pas 1500. Il correspond au modèle
`checkpoint-1500` conservé dans le processus de sauvegarde avant que
les évaluations ultérieures ne remontent.

La courbe complète est disponible dans `results/training_curves.png`.

### 5.4 Décision

La décision technique enregistrée est :

> CONSERVER : gain moyen de WER +0.215, seuil 0.005, aucun jeu de test pire
> de plus de 0.01.

Le texte exact est disponible dans
`results/asr_finetune_decision.txt`.

## 6. Exemple de prédictions

Les prédictions brutes du modèle ajusté montrent des erreurs lexicales
typiques : mots Wolof proches, segmentation des mots, accents, ponctuation
et expressions longuement répétées.

Exemples issus du fichier `results/predictions_whisper-wolof-ft_test_hf.csv` :

| Référence | Prédiction |
|---|---|
| dama lay jéndal piis bi nga fey ñaw bi | dama lay jëndal piis bi nga fay ñaw bi |
| am na tefes yu bare ... | amna ay tefes yu bare ... |
| xëy na yoon yi mën nañoo uute | xëy naa yoon yi mën nañu a ute |

Ces exemples ne doivent pas être interprétés comme des phrases médicales
validées. Ils illustrent uniquement les erreurs du modèle ASR.

## 7. Déploiement

### 7.1 Intégration prévue

Le dossier du modèle final à déployer est annoncé par Kaggle :

`/kaggle/working/whisper-wolof-ft/final`

Dans le dépôt local, le chemin prévu est :

`checkpoints/whisper-wolof-ft/final`

L'application chargera ce dossier lors de la reconnaissance vocale
lorsqu'il sera présent.

### 7.2 Extraction à faire

Le modèle déployable doit contenir au minimum :

- les poids `model.safetensors` ou `pytorch_model.bin` ;
- `config.json` ;
- `preprocessor_config.json` ;
- les fichiers de tokenizer ;
- les métadonnées Whisper nécessaires au chargement.

Le dossier téléchargé depuis Kaggle doit être placé dans
`checkpoints/whisper-wolof-ft/final/`.

Le script `scripts/prepare_deployed_model.py` vérifie ensuite :

- que l'archive est un ZIP ;
- que les poids sont présents ;
- que le chemin final peut être chargé ;
- qu'il ne s'agit pas d'un fichier `.crdownload` incomplet.

### 7.3 État du déploiement

Les résultats Kaggle sont intégrés au dépôt.

Les poids du modèle ne sont pas encore présents localement dans
`checkpoints/whisper-wolof-ft/final`.

Le dossier `faadjou_results (1)` contient les rapports, prédictions,
statistiques et courbes. Il ne contient pas les poids ASR.

Pour cette raison, l'application reste en mode modèle non installé
jusqu'à la récupération complète du dossier final Kaggle.

## 8. Limites et recommandations

### 8.1 Limites techniques

- WER test de 0,5194 : environ une erreur sur deux mots en moyenne ;
- aucun jeu médical Wolof indépendant documenté ;
- traduction Wolof vers français non mesurée dans ces résultats ;
- aucune mesure sur les locuteurs, accents ou bruit ambiant ;
- latence locale dépendante du matériel ;
- archive du modèle à récupérer avant test de bout en bout.

### 8.2 Validation recommandée

Avant présentation publique, il faut effectuer :

- tests avec des locuteurs sénégalais variés ;
- tests dans des environnements calmes puis bruités ;
- ajout d'un jeu médical séparé ;
- évaluation WER/CER sur ce jeu médical ;
- validation de la traduction Wolof vers français ;
- revue des signaux d'alarme par un médecin ;
- tests du pipeline microphone -> ASR -> traduction -> orientation ;
- comparaison finale entre modèle de base et modèle ajusté.

## 9. Architecture de preuve

Les éléments produits sont organisés ainsi :

```text
results/
  asr_eval.csv
  asr_finetune_compare.csv
  asr_finetune_decision.txt
  best_model.txt
  data_summary.json
  predictions_M9and2M_whisper-small-wolof_test_hf.csv
  predictions_whisper-wolof-ft_test_hf.csv
  run.log
  training_curves.png
  training_log.csv
```

La fiche [MODEL_CARD.md](MODEL_CARD.md) résume les contraintes, usages et
résultats mesurés.

Le guide [KAGGLE_TRAINING.md](KAGGLE_TRAINING.md) décrit la préparation et
l'entraînement.

## 10. Conclusion

L'entraînement Kaggle v2 a montré une amélioration nette du modèle Wolof
sur le jeu de test mesuré : le WER passe de 0,7346 à 0,5194 et le CER passe
de 0,4767 à 0,2935. Le meilleur modèle de validation est obtenu au pas 1500,
avec un WER de 0,4134.

La méthode limite les risques de recoupement entre entraînement et
évaluation, conserve le meilleur point de validation et fournit des fichiers
comparables au jury.

Le gain justifie de conserver le modèle ajusté pour les prochains essais.
Il ne suffit pas, à lui seul, à certifier son usage médical : le pipeline
doit être complété par un jeu médical indépendant, une validation clinique
et un déploiement vérifié des poids.
