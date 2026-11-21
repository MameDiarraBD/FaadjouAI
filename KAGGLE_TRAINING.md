# Entraîner Faadjou Wax sur Kaggle

## Résultat attendu

Ce guide produit un modèle Whisper Wolof fine-tuné à partir des deux corpus
Wolof déjà présents sur votre ordinateur. Kaggle fait uniquement
l’entraînement : il ne faut pas charger des gigaoctets de corpus complets.

## 1. Préparer les fichiers sur Windows

Depuis la racine `faadjou-ai`, lancez :

```powershell
python scripts\prepare_subset.py --config configs\config.yaml
```

Le script doit créer ces fichiers :

```text
data\subset\
  train.parquet
  test.parquet
  MANIFEST.json
  medical_test\           (absent pour l'instant si vous n'avez pas fourni de données)
```

Pour ce projet, `MANIFEST.json` confirme notamment :

- `n_train` : nombre d’exemples d’entraînement préparés
- `n_test` : nombre d’exemples de test
- `under_budget` : doit valoir `true`
- `seed` : graine reproductible, actuellement `42`

Ne chargez pas les dossiers `wolof dataset` ni `wolof-french dataset` entiers
sur Kaggle : ils sont déjà combinés sous forme d’échantillon compact par
`prepare_subset.py`.

## 2. Créer deux datasets Kaggle

### Dataset A — code

Sur Kaggle : `Datasets > New Dataset`.

Chargez tout le dossier `faadjou-ai`, sauf les gros dossiers optionnels :

```text
app.py
README.md
requirements.txt
pyproject.toml
configs\
scripts\
src\
tests\
rules\
notebooks\
```

Sans nécessairement les charger :

```text
.venv\
__pycache__\
checkpoints\
results\
data\subset\       (laissez ce dossier dans Dataset B)
```

Nommez ce dataset `faadjou-code`.

### Dataset B — données audio

Chargez uniquement le contenu de :

```text
faadjou-ai\data\subset\
```

Le dossier Kaggle contiendra donc au minimum :

```text
train.parquet
test.parquet
MANIFEST.json
```

Nommez ce dataset `faadjou-wolof-subset`.

## 3. Ouvrir le notebook

Dans Kaggle : `Code > New Notebook`.

Dans les paramètres du notebook :

- GPU : `GPU T4 x2` si disponible, ou `T4 x1`
- Internet : activé
- Ajouter le dataset `faadjou-code`
- Ajouter le dataset `faadjou-wolof-subset`

## Si PyPI ou Hugging Face dit « ERREUR DNS »

L’erreur signifie que la session Kaggle est sans accès réseau. Dans
`Notebook options > Internet`, choisissez **On** avant toute exécution.
Redémarrez la session après activation, puis relancez la cellule d’installation.

Dans Kaggle, vérifiez aussi :

```python
import socket
print(socket.gethostbyname("pypi.org"))
print(socket.gethostbyname("huggingface.co"))
```

Si ces commandes échouent, ne lancez pas l’entraînement : aucun modèle ni paquet
ne peut être téléchargé tant que le DNS ne répond pas. Essayez un redémarrage
de la session avec Internet activé, puis vérifiez à nouveau.

Téléversez ensuite le fichier
`faadjou-ai/notebooks/kaggle_finetune_whisper.ipynb`,
ou copiez ses cellules.

## 4. Chemins Kaggle attendus

Après ajout des datasets, Kaggle monte généralement les données ici :

```text
/kaggle/input/faadjou-code
/kaggle/input/faadjou-wolof-subset
```

Ne les écrasez pas. Le notebook copie le code vers `/content/faadjou-ai`
et pointe la configuration vers `/kaggle/input/faadjou-wolof-subset`.

## 5. Ordre d’entraînement

La séquence est :

1. Évaluer les modèles ASR Wolof sans entraînement.
2. Vérifier les résultats WER/CER.
3. Fine-tuner Whisper pendant un essai borné à 50 étapes.
4. Évaluer avant/après.
5. Conserver le modèle seulement si le WER test baisse.

Commandes locales et Kaggle équivalentes :

```bash
python scripts/run_asr_eval.py --config configs/config.subset.yaml --max-samples 200 --batch-size 1
python scripts/finetune_asr.py --config configs/config.subset.yaml --max_steps 50 \
  --output_dir /kaggle/working/checkpoints/whisper-wolof-ft
```

Le fine-tuning complet doit venir uniquement après une validation clinique
du jeu de données et des règles d’alarme. Le dossier est repris
automatiquement depuis le dernier checkpoint.

## 6. Télécharger le modèle

Le résultat utile est :

```text
/kaggle/working/checkpoints/whisper-wolof-ft/final
```

Le dossier contient les poids, le processeur et les fichiers du modèle.
Téléchargez-le sous forme de ZIP depuis Kaggle, par exemple :
`faadjou-wolof-model.zip`.

Copiez ensuite son contenu ici :

```text
faadjou-ai/checkpoints/whisper-wolof-ft/final/
```

Puis configurez `config.yaml` ou `config.subset.yaml` :

```yaml
api:
  asr_model: "checkpoints/whisper-wolof-ft/final"
```

Ne déployez pas un modèle si `asr_finetune_decision.txt` indique
« NE PAS CONSERVER ».

## 7. Fichiers de preuve à conserver

```text
results/asr_eval.csv
results/asr_finetune_compare.csv
results/asr_finetune_decision.txt
MANIFEST.json
MODEL_CARD.md
```

Ces fichiers permettent au jury de vérifier les données utilisées,
les mesures réelles et la décision de conserver ou rejeter le modèle.
