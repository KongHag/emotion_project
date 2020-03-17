# Projet 3A : Détection d'émotions dans des séquences de films

## Tâches

### Séance du 23/01/2020

- [x] Finir de lire l'article
- [x] Créer un github
- [x] Envoyer un mail à M. Dellandréa pour les soumissions à [MediaEval](http://www.multimediaeval.org/)
- [X] Explorer la base de données
  
### Séance du 30/01/2020

- [x] Terminer les fonction de lecture des données
- [x] Lecture de doc torch sur les couches RNN
- [x] Premier pipe
- [ ] Premier entraînement
  
### Séance 06/02/2020

- [x] argParser

### Reste à faire

- [ ] Etoffer et mettre à jour le ```README.md```.
- [ ] Créer une fonction score, qui prend en entrée un modèle et qui renvoie son score, indépendament du criterion (Pearson ?), évalué sur le testset.
- [ ] Créer un fichier type carnet d'essais, dans lequel on syntétisera tous les essais, résultats etc, même ceux qui n'ont pas marché.
- [ ] Implémenter un scheduler.
- [x] Implémenter régularisation L2.
- [ ] Implémenter le lstm bidirectionnel (buggé pour l'instant).
- [ ] Ecrire une jolie docstring au début du fichier ```emotion.py```.
- [ ] Ecrire une jolie docstring au début du fichier ```model.py```.
- [ ] Ecrire une jolie docstring au début du fichier ```log.py```.
- [ ] Ecrire une jolie docstring au début du fichier ```training.py```.

--------------------

## Liens utiles

- [LIRIS-ACCEDE](https://liris-accede.ec-lyon.fr/)
- [MediaEval](http://www.multimediaeval.org/)
- [Login ACCEDE](https://liris-accede.ec-lyon.fr/files/database-download/download.php)
- [Thèse Yohan Baveye](https://tel.archives-ouvertes.fr/tel-01272240/document)

--------------------

## Données

### Pour charger le dataset

Télécharger le fichier data.pickle et le mettre dans le dossier ```data```.

```python
from read_data import load_data
x, y = load_data()
```

```data``` est un liste de tuple ```(X, Y)```.

- ```X``` correspond un tenseur (temps, concaténation de tous les features)
- ```Y``` correspond un tenseur (temps, valence/arousal)

Les tenseurs X et Y n'ont pour l'instant pas exactement la même taille dans la dimension du temps.

### Description du dataset

Lien de téléchargement des fichiers : [LIRIS-ACCEDE](https://liris-accede.ec-lyon.fr/)

Il y a quatre sets:

| Set name           | movie id    | Usage    |
| ------------------ | ----------- | -------- |
| ```DevSet-Part1``` | de 0 à 13   | Training |
| ```DevSet-Part2``` | de 14 à 43  | Training |
| ```DevSet-Part3``` | de 44 à 53  | Training |
| ```TestSet```      | de 54 à 65  | Testing  |

Il y a 5 types de donnée :

- Audio features
- Visual features
- Valence and arousal annotations
- Fear annotations
- Data

#### Audio features

Pour récupérer les audio features d'un film, il faut faire :

```python
import read_data

# Get the audio feature  the 12th movie
audio_feature = read_data.audio_feature(12)
```

Consiste en des fichiers ```.csv```. Le path de chaque fichiers est :

```MEDIAEVAL18-{SET_NAME}-Audio_features/audio_features/MEDIAEVAL18_{MOVIE_ID}/MEDIAEVAL18_{MOVIE_ID}_{SEQUENCE_ID}.csv```

Avec

- ```SET_NAME``` le nom du set (```DevSet-Part1```, ```DevSet-Part2```, ```DevSet-Part3``` ou ```TestSet```).
- ```MOVIE_ID``` id du film, sur 2 caractères.
- ```SEQUENCE_ID``` id du la séquence audio, sur 5 caractères.

Il y a un fichier par seconde de film. Les descripteurs sont évaluées sur une fenêtre de 5 secondes. L'identifiant de la partition correspond à la seconde au centre de la séquence de cinq secondes.

Exemple : le fichier ```MEDIAEVAL18_19_00123.csv``` donne les descripteurs audios du film 19, pour la séquence audio comprise entre 121.5 sec et 125.5 sec.

Cas particulier des séquences de début et de fin : les descripteurs des indices ```00001``` et ```00002``` sont une copie des descripteurs de la séquence ```00003```.

Pour chaque image, il y a 1 582 descripteurs :

- 1 428 descripteurs type LLD (21 fonctions appliquées sur 68 contours LLD (low-level_descriptor))
- 152 descripteurs type pitch-based LLD (19 fonctions appliquées sur les 8 pitch-based LLD)
- 2 descripteurs : "the number of pitch onsets (pseudo syllables) and the total duration of the input are appended"

#### Visual features

Pour récupérer une visual feature d'un film, il suffit d'écrire :

```python
import read_data

# Get the audio feature of the 12th movie
eh_feature = read_data.visual_feature(12, "eh")
```

Les descripteurs sont les suivants :

- Auto Color Correlogram (acc)
- Color and Edge Directivity Descriptor (cedd)
- Color Layout (cl)
- Edge Histogram (eh)
- Fuzzy Color and Texture Histogram (fcth)
- Gabor (gabor)
- Joint descriptor joining CEDD and FCTH in one histogram (jcd)
- Scalable Color (sc)
- Tamura (tamura)
- Local Binary Patterns (lbp)
- VGG16 fc6 layer output (fc6)

Chaque seconde, une frame est extraite du film. Pour cette frame extraite, tous les descripteurs sont évalués. Il y a un ficher ```.txt``` par frame et par descripteur. Le path est :

```MEDIAEVAL18-{SET_NAME}-Visual_features/visual_features/MEDIAEVAL18_{MOVIE_ID}/{FEATURE}/MEDIAEVAL18_{MOVIE_ID}_{IMAGE_ID}_{FEATURE}.txt```

Avec

- ```SET_NAME``` : le nom du set (```DevSet-Part1```, ```DevSet-Part2```, ```DevSet-Part3``` ou ```TestSet```).
- ```MOVIE_ID``` : id du film, sur 2 caractères, de 0 à 65.
- ```IMAGE_ID``` : id du la frame, sur 5 caractères.
- ```FEATURE``` : nom abrégé du descripteur (```acc```, ```cedd```, ```cl```, ```eh```, ```fcth```, ```gabor```, ```icd```, ```sc```, ```tamura```, ```lbp``` ou ```fc6```).

Le fichier contient une liste de valeurs décimales séparées par une ```,``` dont la taille dépend du descripteur.

### Valence and arousal annotation

Pour récupérer les valeurs valence et arousal d'un film, il suffit d'écrire :

```python
import read_data

# Get the valence and arousal of the 12th movie
valence_arousal = read_data.valence_arousal(12)
```

Pour chaque seconde, une valeur de valence et d'arousal sont stockées dans un fichier ```.txt```.
Le path est :

```MEDIAEVAL18-{SET_NAME}-Valence-Arousal/annotation/MEDIAEVAL18_{MOVIE_ID}_Valence-Arousal.txt```

Avec

- ```SET_NAME``` : le nom du set (```DevSet-Part1```, ```DevSet-Part2```, ```DevSet-Part3``` ou ```TestSet```).
- ```MOVIE_ID``` : id du film, sur 2 caractères, de 0 à 65.

Le fichier contient une liste de valeurs décimales séparées par une ```,``` . Les colonnes sont dans l'ordre : le temps, la valence et l'arousal.

### Fear annotation

Pas encore utilisé

### Data

Pas encore utilisé

--------------------

## Fonctionnement des scripts

Fonctionnement du argparser :

```bash
usage: emotion.py [-h] [--seq-len SEQ_LEN] [--num-hidden NUM_HIDDEN]
                  [--hidden-size HIDDEN_SIZE] [--lr LR]
                  [--batch-size BATCH_SIZE] [--grad-clip GRAD_CLIP]
                  [--nb-epoch NB_EPOCH] [-O {Adam,RMSprop,SGD}]
                  [-C {MSE,Pearson}] [--weight-decay WEIGHT_DECAY]
                  [-D DROPOUT] [--logger-level LOGGER_LEVEL]
                  [--fragment FRAGMENT]

Train Neural Network for emotion predictions

optional arguments:
  -h, --help            show this help message and exit
  --seq-len SEQ_LEN     Length of a sequence
  --num-hidden NUM_HIDDEN
                        Number of hidden layers in NN
  --hidden-size HIDDEN_SIZE
                        Dimension of hidden layer
  --lr LR               Learning rate
  --batch-size BATCH_SIZE
                        Size of a batch
  --grad-clip GRAD_CLIP
                        Boundaries of the gradient clipping function (centered
                        on 0)
  --nb-epoch NB_EPOCH   Number of epoch
  -O {Adam,RMSprop,SGD}, --optimizer {Adam,RMSprop,SGD}
                        Type of optimizer
  -C {MSE,Pearson}, --crit {MSE,Pearson}
                        Typer of criterion for loss computation
  --weight-decay WEIGHT_DECAY
                        L2 regularizations coefficient
  -D DROPOUT, --dropout DROPOUT
                        Dropout probability between [0, 1]
  --logger-level LOGGER_LEVEL
                        Logger level: from 10 (debug) to 50 (critical)
  --fragment FRAGMENT   The percentage of the dataset used. From 0 to 1
quentingallouedec@MacBook-Pro-de-Quentin emotion_project % 
```
