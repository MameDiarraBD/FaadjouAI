# Données audio locales

## Corpus général

Placez vos fichiers ici et un `metadata.csv` :

```text
data/audio/
  metadata.csv
  wavs/
    001.wav
    002.wav
```

`metadata.csv` :

```csv
file,transcription
wavs/001.wav,maangi feebar
wavs/002.wav,sama bopp dafa metti
```

Puis dans `configs/config.yaml` :

```yaml
audio:
  source:
    type: local
    local_dir: data/audio
    metadata_csv: metadata.csv
```

## Jeu medical_test (séparé, jamais en train)

```text
data/audio/medical_test/
  metadata.csv
  wavs/...
```

## Sous-échantillonnage rapide

```yaml
audio:
  max_samples: 200
```
