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

## Limites

- Traduction Wolof elle-même non validée cliniquement.
- Accent, environnement bruité, âge et dialecte peuvent dégrader la transcription.
- Toute sortie doit être relue avant usage réel.
- Les règles d'alarme restent marquées « À VALIDER PAR UN MÉDECIN ».
