# Projet 3A : Détection d'émotions dans des séquences de films
# Tâches
## Séance du 23/01/2020
- [x] Finir de lire l'article
- [x] Créer un github
- [ ] Envoyer un mail à M. Dellandréa pour les soumissions à [MediaEval](http://www.multimediaeval.org/)
- [ ] Explorer la base de données
## Séance du 30/01/2020
- [ ] Lecture de doc torch sur les couches RNN
- [ ] Premier pipe
- [ ] Premier entraînement
--------------------
## Liens utiles
+ [LIRIS-ACCEDE](https://liris-accede.ec-lyon.fr/)
+ [MediaEval](http://www.multimediaeval.org/)
+ [Login ACCEDE](https://liris-accede.ec-lyon.fr/files/database-download/download.php)
+ [Thèse Yohan Baveye](https://tel.archives-ouvertes.fr/tel-01272240/document)
--------------------
## Données
Lien de téléchargement des fichiers : + [LIRIS-ACCEDE](https://liris-accede.ec-lyon.fr/)

Il y a quatre batches:
| Nom du batch | id film     | Usage    |
|--------------|-------------|----------|
| DevSet Part1 | de 0 à 13   | Training |
| DevSet Part2 | de 14 à 43  | Training |
| DevSet Part3 | de 44 à 53  | Training |
| DevSet Part1 | de 54 à 65  | Testing  |

Il y a 5 types de données : 
+ Audio feature
+ Visual feature
+ Valence and arousal annotation
+ Fear annotation
+ Data


### Audio features

Consiste en des fichiers ```.csv```. Le path de chaque fichiers est :
```
MEDIAEVAL18-DevSet-{BATCH_NAME}-Audio_features/audio_features/MEDIAEVAL18_{FILM_ID}/MEDIAEVAL18_{FILM_ID}_{PARTITION_ID}.csv
```
Avec 
+ ```BATCHNAME``` le nom de batch (```Part1```, ```Part2```, ```Part3``` ou ```TestSet```).
+ ```FILM_ID``` id du film, sur 2 caractères.
+ ```PARTITION_ID``` id du la partition audio, sur 5 caractères.

Il y a un fichier par seconde de film. Les descripteurs sont évaluées sur une fenêtre de 5 secondes. L'identifiant de la partition correspond à la seconde au centre de la séquence de cinq secondes. 

Exemple : le fichier ```MEDIAEVAL18_19_00123.csv``` donne les descripteurs audios du film 19, pour la séquence audio comprise entre 121.5 sec et 125.5 sec.

Cas particulier des séquence de début et de fin : les descripteurs des indices ```00001``` et ```00002``` sont une copie des descripteurs de la séquence ```00003````.

A COMPLETER : CONTENU DES FICHIER ```.csv```


### Visual features

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

Chaque seconde, une frame est extraite du film. Pour cette frame extraite, tous les descripteurs sont évalués. Il y a un ficher ```.txt```par frame et par descripteur. Le path est : 

```
MEDIAEVAL18-DevSet-{BATCH_NAME}-Visual_features/visual_features/MEDIAEVAL18_{FILM_ID}/{DESCRIPTOR}/MEDIAEVAL18_{FILM_ID}_{PARTITION_ID}_{DESCRIPTOR}.txt
```
Avec 
+ ```BATCHNAME``` : le nom de batch (```Part1```, ```Part2```, ```Part3``` ou ```TestSet```).
+ ```FILM_ID``` : id du film, sur 2 caractères, de 0 à 65.
+ ```PARTITION_ID``` : id du la partition audio, sur 5 caractères.
+ ```DESCRIPTOR ``` : nom abrégé du descripteur (```acc```, ```cedd```, ```cl```, ```eh```, ```fcth```, ```gabor```, ```icd```, ```sc```, ```tamura```, ```lbp``` ou ```fc6```).

Le fichier contient une liste de valeurs décimales séparées par une ```,``` dont la taille dépend du descripteur.


### Valence and arousal annotation

A COMPLETER

### Fear annotation

A COMPLETER

### Data

A COMPLETER


## Fonctionnement des scripts
Description des fonctionnements des scripts
