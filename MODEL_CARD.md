# Carte du modèle ASR Wolof — Faadjou

## Usage prévu

Reconnaissance vocale Wolof pour soutenir l'orientation médicale de Faadjou.
Le modèle ne pose pas de diagnostic et ne remplace pas un professionnel de santé.

## Jeux de données

- Corpus Wolof local configuré dans `configs/config.yaml`.
- Corpus Wolof-français local, filtré sur la langue Wolof.
- Jeu médical séparé, jamais inclus dans l'entraînement.

## Entraînement

Le fine-tuning est préparé pour Kaggle avec Whisper small, graines reproductibles,
batch/accumulation bornés et reprise depuis le dernier checkpoint.

## Évaluation

WER, CER, latence et jeu médical sont calculés par les scripts existants.
Le fichier `asr_finetune_decision.txt` décide de conserver le modèle uniquement
si le WER test s'améliore.

## Résultat Kaggle mesuré

- Modèle de base : WER `0.7346`, CER `0.4767`, sur 300 exemples Wolof.
- Modèle fine-tuné : WER `0.5194`, CER `0.2935`, sur les mêmes 300 exemples.
- Amélioration absolue du WER : `0.2152`.
- Meilleur WER de validation : `0.4134`, obtenu au pas `1500`.
- Décision Kaggle : conserver le modèle fine-tuné.

Ces valeurs proviennent de `results/asr_finetune_compare.csv`.

## Limites

- Traduction Wolof elle-même non validée cliniquement.
- Accent, environnement bruité, âge et dialecte peuvent dégrader la transcription.
- Toute sortie doit être relue avant usage réel.
- Les règles d'alarme restent marquées « À VALIDER PAR UN MÉDECIN ».
