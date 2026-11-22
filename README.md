# Faadjou AI

Plateforme de mise en relation **patients ↔ professionnels de santé** (concours IA Sénégal).

Deux modules :

| Module | Rôle |
|--------|------|
| **Faadjou Wax** | Triage vocal en wolof : ASR → traduction → orientation JSON (spécialité + urgence) |
| **Sentinelle** | Alerte précoce climat-santé sur **événements SIMULÉS** corrélés à la pluie |

> **Avertissement** : ce système **oriente**, il **ne pose jamais de diagnostic** et **ne remplace pas un médecin**. Les listes de signaux d’alarme et de spécialités sont des **ébauches à faire relire par un médecin**.

---

## Objectif

Fournir un pipeline **reproductible** (local + Google Colab T4 gratuit) qui :

1. Compare des ASR wolof existants (WER / CER) sans inventer de chiffres.
2. Permet un fine-tuning Whisper *small* optionnel, avec reprise après coupure de session.
3. Oriente via un LLM (local 4-bit ou API) **sous contrôle de règles explicites** (red flags).
4. Détecte des pics SIMULÉS liés à la pluie (Open-Meteo) pour démontrer Sentinelle.

**Pourquoi ces choix techniques (défense jury)**  
- Config unique `configs/config.yaml` : reproductibilité.  
- Graine fixée (`project.seed`) : splits et simulation stables.  
- Red flags **après** le LLM : le modèle ne décide jamais seul d’un cas grave.  
- Pas de clés dans le code : `HF_TOKEN`, etc. via l’environnement / Secrets Colab.  
- `torch` non épinglé : Colab le fournit ; en local, installer selon votre CUDA.

---

## Installation locale

```bash
cd faadjou-ai
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
# source .venv/bin/activate

# PyTorch (adapter à votre machine) : https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

pip install -r requirements.txt
pip install -e .

# Secrets (optionnel)
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
```

Tests unitaires (sans GPU, sans modèles lourds) :

```bash
pytest -q
```

---

## Petit paquet de données (à faire avant Colab, Kaggle ou Hugging Face)

Les dossiers complets pèsent plusieurs gigaoctets. On ne les envoie pas.
Depuis la racine du projet, là où les chemins `../wolof dataset` et
`../wolof-french dataset` de `configs/config.yaml` sont valides :

```bash
python scripts/prepare_subset.py --config configs/config.yaml
```

Cela écrit `data/subset/` :

- `train.parquet` et `test.parquet` (environ 300 extraits, test issu seulement des fichiers `test-*`)
- `medical_test/` (vos phrases, si `data/audio/medical_test/metadata.csv` existe)
- `MANIFEST.json` (compteurs et taille, sans chemin absolu)

Le plafond est `subset.max_total_mb` (190). Ensuite tous les scripts audio
utilisent `configs/config.subset.yaml`.

### Google Drive

1. Déposez le dossier `data/subset` dans Drive (glisser-déposer, pas les corpus complets).
2. Déposez aussi le code `faadjou-ai`.
3. Dans Colab :

```python
from google.colab import drive
drive.mount("/content/drive")
# Lien relatif : le paquet est là où vous l'avez mis, par exemple
# /content/drive/MyDrive/faadjou-subset
# Copiez-le à côté du code, ou éditez configs/config.subset.yaml :
#   audio.source.parquet_sources[0].path = "/content/drive/MyDrive/faadjou-subset"
```

### Kaggle

1. New Dataset → uploadez le contenu de `data/subset` (les deux parquet, `MANIFEST.json`, `medical_test/`).
2. Dans le notebook, le chemin est `/kaggle/input/<nom-du-dataset>`.
3. Mettez ce chemin dans `audio.source.parquet_sources[0].path` et
   `audio.medical_test_dir` de `configs/config.subset.yaml`.

### Hugging Face (dataset privé)

```bash
# HF_TOKEN dans l'environnement, jamais dans un fichier suivi par git
huggingface-cli login
huggingface-cli upload VOTRE_COMPTE/faadjou-wolof-subset data/subset . --repo-type dataset --private
```

Puis dans la config : `audio.source.type: huggingface` et
`audio.source.huggingface_id: VOTRE_COMPTE/faadjou-wolof-subset`.
Le jeu Hugging Face doit garder les fichiers `train.parquet` et `test.parquet`.

## Ordre d'exécution

Le fine-tuning est la dernière étape, et il est optionnel.

```bash
# 1. ASR sans entraînement (200 exemples, CPU ou petit GPU)
python scripts/run_asr_eval.py --config configs/config.subset.yaml --max-samples 200 --batch-size 1

# 2. Triage (scénarios français → JSON → règles)
python scripts/run_triage_eval.py --config configs/config.yaml

# 3. Sentinelle (simulation + Open-Meteo + graphique)
python scripts/run_sentinelle.py --config configs/config.yaml

# 4. API pour Spring Boot
python scripts/serve_api.py --config configs/config.yaml
# GET  /health    POST /triage

# 5. En dernier seulement : fine-tuning borné
python scripts/finetune_asr.py --config configs/config.subset.yaml --max_steps 50
```

Sur Colab ou Kaggle, les mêmes commandes, après avoir pointé `config.subset.yaml`
vers le paquet. Si la session se coupe pendant l'étape 5, relancez la même
commande : reprise au dernier checkpoint.

WER et CER ne sont écrits dans `results/asr_eval.csv` que par le script 1.
Ce dépôt ne contient pas de scores inventés. Sans vos enregistrements médicaux,
la ligne `medical_test` a `n_exemples = 0` et une erreur explicite.

## Format des données audio

Voir aussi `data/audio/README.md`.

**Option A — Hugging Face** (`configs/config.yaml`) :

```yaml
audio:
  source:
    type: huggingface
    huggingface_id: "VOTRE_DATASET_WOLOF"
    huggingface_split: train
```

Le dataset doit exposer une colonne `audio` et une transcription (`transcription`, `text` ou `sentence`).

**Option B — Dossier local** :

```text
data/audio/metadata.csv   # colonnes : file, transcription
data/audio/wavs/*.wav
```

```yaml
audio:
  source:
    type: local
    local_dir: data/audio
```

**Jeu `medical_test`** (phrases médicales, **jamais** en entraînement) :

```text
data/audio/medical_test/metadata.csv
data/audio/medical_test/wavs/...
```

Sous-échantillonnage : `--max-samples 200` lit au plus 200 extraits de train et 200 de test, sans ouvrir tout le corpus.

**Scénarios triage** : `tests/scenarios.csv` (parti du modèle `tests/scenarios_template.csv`).

Colonnes : `id,texte_fr,audio_wolof_optionnel,specialite_attendue,urgence_attendue`.

---

## Colab (GPU T4), après le petit paquet

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/faadjou-ai
!pip install -q -r requirements.txt
!pip install -q -e .
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

!python scripts/run_asr_eval.py --config configs/config.subset.yaml --max-samples 200 --batch-size 1
!python scripts/run_triage_eval.py --config configs/config.yaml
!python scripts/run_sentinelle.py --config configs/config.yaml
# En dernier seulement :
!python scripts/finetune_asr.py --config configs/config.subset.yaml --max_steps 50 \
  --output_dir /content/drive/MyDrive/faadjou-ai/checkpoints/whisper-wolof-ft
```

Le dossier Drive reçoit les checkpoints. Relancer la même commande reprend le dernier checkpoint.
`CAYTU/whosper-large-v2` reste commenté dans la config : trop grand pour un T4 16 Go en général.

---

## Comment lire `results/`

| Fichier | Contenu |
|---------|---------|
| `asr_eval.csv` | Par modèle et jeu (`test` / `medical_test`) : WER, CER, n, latence moy. |
| `asr_finetune_compare.csv` | WER/CER **avant** vs **après** fine-tuning + décision de conservation |
| `asr_finetune_decision.txt` | Message clair : conserver ou non le modèle ajusté |
| `triage_eval.csv` | Accuracy spécialité, **rappel urgences vitales**, taux JSON, latence |
| `triage_errors.csv` | Détail par scénario + urgences vitales manquées |
| `sentinelle_events_SIMULE.csv` | Événements **SIMULÉS** anonymes |
| `sentinelle_injected_spikes_SIMULE.json` | Pics injectés (vérité terrain pour l’éval) |
| `sentinelle_eval.csv` | Pics détectés, délai moyen, fausses alertes (**données SIMULÉES**) |
| `sentinelle_pic.png` | Graphique d’un pic + pluie (annoté SIMULÉ) |
| `sentinelle_weather_*.csv` | Météo Open-Meteo (attribution dans les colonnes / logs) |

**Aucun chiffre n’est prérempli dans ce dépôt** : ce sont les scripts qui écrivent les résultats après exécution.

---

## Tableau des résultats à remplir (après vos runs)

| Expérience | Métrique | Valeur mesurée | Commentaire |
|------------|----------|----------------|-------------|
| ASR modèle 1 / test | WER | _à remplir_ | |
| ASR modèle 1 / medical_test | WER | _à remplir_ | |
| ASR modèle 2 / test | WER | _à remplir_ | |
| ASR après fine-tuning / test | WER | _à remplir_ | conserver ? oui/non |
| Triage | Accuracy spécialité | _à remplir_ | |
| Triage | Rappel urgences vitales | _à remplir_ | **prioritaire** |
| Triage | Taux JSON valide | _à remplir_ | |
| Sentinelle | Taux pics injectés détectés | _à remplir_ | SIMULÉ |
| Sentinelle | Délai moyen (jours) | _à remplir_ | SIMULÉ |
| Sentinelle | Fausses alertes | _à remplir_ | SIMULÉ |

---

## Architecture rapide

```text
Audio wolof → ASR → Texte wolof → NLLB (wol_Latn→fra_Latn)
    → LLM (JSON) → Validation jsonschema → Règles red_flags → Orientation
```

Sentinelle :

```text
Simulation événements (+ pics après pluie) + Open-Meteo
    → comptages → moyenne mobile + k·σ → alertes → éval / PNG
```

---

## Limites connues

- Qualité ASR wolof **dépend des données** ; sans corpus local / HF renseigné, l’éval ASR ne peut pas tourner.
- Le LLM local 4-bit peut être lent ou approximatif ; basculez `llm.backend: api` si besoin.
- `red_flags.json` et `specialites.json` : **non validés cliniquement**.
- Scénarios du template : **fictifs**.
- Sentinelle démontre la **méthode** sur données **SIMULÉES**, pas une surveillance réelle.
- Sessions Colab limitées en temps / disque : utilisez Drive + `save_total_limit`.
- Licence LLM par défaut : **Qwen2.5-1.5B-Instruct (Apache-2.0)** — revérifier avant soumission si vous changez de modèle.

---

## À faire valider par un humain (checklist jury)

- [ ] Liste des **signaux d’alarme** (`rules/red_flags.json`) — marquage *NON VALIDÉE*
- [ ] Liste des **spécialités** (`rules/specialites.json`)
- [ ] **Numéro d’urgence** (`triage.emergency_phone` dans la config)
- [ ] Scénarios cliniques (`tests/scenarios.csv`)
- [ ] Enregistrements **medical_test**
- [ ] Choix du meilleur checkpoint ASR après `asr_eval.csv`
- [ ] Décision de conservation du modèle fine-tuné (`asr_finetune_decision.txt`)

---

## Licence / éthique

Usage concours / démonstration. Pas un dispositif médical certifié.  
Météo : **Open-Meteo** (https://open-meteo.com/).

## Lancer la démo Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

L’application mobile d’abord présente trois espaces : Faadjou Wax (démo d’orientation),
Sentinelle (simulation sanitaire), et Rendez-vous (praticiens fictifs, sans paiement).
Le Wolof n’est pas généré par le prototype : les zones audio affichent `TODO_WOLOF`
et proposent un repli français explicitement identifié « wolof à venir ».

## Préparer les modèles Wolof sur Kaggle

Voir aussi le guide détaillé : [KAGGLE_TRAINING.md](KAGGLE_TRAINING.md).

Pour le jury : [DOCUMENT_TECHNIQUE.md](DOCUMENT_TECHNIQUE.md) reprend les
données, la méthode, les métriques mesurées, les limites et le protocole de
déploiement du modèle ASR.

1. Exécutez `python scripts/prepare_subset.py --config configs/config.yaml`.
2. Le paquet mélange les deux corpus configurés, conserve les splits test,
   borne la taille et écrit `data/subset/MANIFEST.json`.
3. Téléversez `faadjou-ai` et `data/subset` comme datasets Kaggle séparés.
4. Lancez [le notebook Kaggle](/notebooks/kaggle_finetune_whisper.ipynb).
5. Téléchargez `faadjou-wolof-model.zip`, puis placez le modèle dans `checkpoints/`
   avant branchement à l’application.

Le notebook effectue une évaluation baseline, un essai borné à 50 étapes,
une évaluation après entraînement et écrit la décision de conservation.

## Déployer le modèle Kaggle obtenu

Après téléchargement complet de `faadjou_model_ft.zip`, vérifiez que Chrome a
produit un vrai fichier `.zip` terminé à 100 % (pas un `.crdownload`).
Placez-le à la racine du projet puis lancez :

```powershell
python scripts\prepare_deployed_model.py faadjou_model_ft.zip
```

Le script vérifie l’intégrité ZIP, extrait les poids et copie le dossier final
dans `checkpoints/whisper-wolof-ft/final/`.

Le modèle Kaggle actuel a été mesuré à WER `0.5194` contre `0.7346` pour le
modèle de base sur 300 exemples. Les détails mesurés sont conservés dans
`results/asr_finetune_compare.csv` et `MODEL_CARD.md`.
