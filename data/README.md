# MediaEval18

## Charger les données

Après avoir généré les données, on peut charger le dataset en exectuant les commandes suivantes.

```python
from read_data import load_data
X, Y = load_data()
```

```X```et ```Y``` sont des listes. Chaque élement de ces listes correspond à un film, et sont des arrays numpy. Les dimensions pour ces arrays sont :

- ```(time, features)``` pour ```X```.
- ```(time, valence_arousal)``` pour ```Y```.

Une description complète du des données se trouve dans le ```README.md``` du dossier ```data```.

## Composition du dataset

Lien de téléchargement des fichiers : [LIRIS-ACCEDE](https://liris-accede.ec-lyon.fr/)

Il y a quatre sets:

| Set name           | movie id    | Usage    |
| ------------------ | ----------- | -------- |
| ```DevSet-Part1``` | de 0 à 13   | Training |
| ```DevSet-Part2``` | de 14 à 43  | Training |
| ```DevSet-Part3``` | de 44 à 53  | Training |
| ```TestSet```      | de 54 à 65  | Testing  |

Il y a 5 types de donnée :

En entrée :
- Audio features
- Visual features
- Data
  
En sortie :
- Valence and arousal annotations (sortie)
- Fear annotations (sortie)

--------------------

## Audio features

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

--------------------

## Visual features

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

--------------------

## Valence and arousal annotation

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
