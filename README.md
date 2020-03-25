# Projet 3A : Détection d'émotions dans des séquences de films

## Utilisation

### Générer les données

Télécharger l'ensemble des fichiers de MediaEval2018: [LIRIS-ACCEDE](https://liris-accede.ec-lyon.fr/), et placez-les dans le dossier ```data```de ce répertoire.

Générer les fichiers ```x_train.pickle```, ```y_train.pickle```, ```x_mean.pickle``` et ```x_std.pickle``` et executant les commandes suivantes:

```python
from read_data import dump_data
dump_data()
```

### Lancer un entrainement

Pour lancer un entrainement, il faut executer le script ```emotion.py```.

```
usage: emotion.py [-h] [--add-CNN] [--seq-len SEQ_LEN]
                  [--num-hidden NUM_HIDDEN] [--hidden-size HIDDEN_SIZE]
                  [--lr LR] [--batch-size BATCH_SIZE] [--grad-clip GRAD_CLIP]
                  [--nb-epoch NB_EPOCH] [-O {Adam,RMSprop,SGD}] [-B BIDIRECT]
                  [--weight-decay WEIGHT_DECAY] [-D DROPOUT]
                  [--logger-level LOGGER_LEVEL] [--fragment FRAGMENT]
                  [--features {acc,cedd,cl,eh,fcth,gabor,jcd,sc,tamura,lbp,fc6,visual,audio,all} [{acc,cedd,cl,eh,fcth,gabor,jcd,sc,tamura,lbp,fc6,visual,audio,all} ...]]
                  [--no-overlapping]

Train Neural Network for emotion predictions

optional arguments:
  -h, --help            show this help message and exit
  --add-CNN             Use the model with a first layer of CNNs
  --seq-len SEQ_LEN     Length of a sequence
  --num-hidden NUM_HIDDEN
                        Number of hidden layers in NN
  --hidden-size HIDDEN_SIZE
                        Dimension of hidden layer
  --lr LR               Learning rate
  --batch-size BATCH_SIZE
                        Size of a batch
  --grad-clip GRAD_CLIP
                        Gradient clipped between [- grad-clip, grad-clip]
  --nb-epoch NB_EPOCH   Number of epoch
  -O {Adam,RMSprop,SGD}, --optimizer {Adam,RMSprop,SGD}
                        Type of optimizer
  -B BIDIRECT, --bidirect BIDIRECT
                        Whether to use bidirectional
  --weight-decay WEIGHT_DECAY
                        L2 regularization coefficient
  -D DROPOUT, --dropout DROPOUT
                        Dropout probability between [0, 1]
  --logger-level LOGGER_LEVEL
                        Logger level: from 10 (debug) to 50 (critical)
  --fragment FRAGMENT   The percentage of the dataset used. From 0 to 1
  --features {acc,cedd,cl,eh,fcth,gabor,jcd,sc,tamura,lbp,fc6,visual,audio,all} [{acc,cedd,cl,eh,fcth,gabor,jcd,sc,tamura,lbp,fc6,visual,audio,all} ...]
                        Features used
  --no-overlapping      Forbid overlapping between sequences in dataset
```

La configuration choisie, ainsi que les résultats seront stockés dans le dossier ```results``` sous le format ```json```.

--------------------

## Tâches pour le projet

### Séance du 23/01/2020

- [x] Finir de lire l'article
- [x] Créer un github
- [x] Envoyer un mail à M. Dellandréa pour les soumissions à [MediaEval](http://www.multimediaeval.org/)
- [X] Explorer la base de données
  
### Séance du 30/01/2020

- [x] Terminer les fonction de lecture des données
- [x] Lecture de doc torch sur les couches RNN
- [x] Premier pipe
- [x] Premier entraînement
  
### Séance 06/02/2020

- [x] argParser

### Reste à faire

- [ ] Etoffer et mettre à jour le ```README.md```.
- [ ] Créer une fonction score, qui prend en entrée un modèle et qui renvoie son score, indépendament du criterion (Pearson ?), évalué sur le testset.
- [x] Création d'un dossier ```results``` dans lequel est stocké un historique des configuration et les résultats obtenus
- [x] Fonction qui lit les fichier du dossier ```results```et met en forme les résultats obtenus.
- [ ] Implémenter un scheduler.
- [x] Implémenter régularisation L2.
- [x] Implémenter le lstm bidirectionnel (buggé pour l'instant).
- [x] Ecrire une jolie docstring au début du fichier ```emotion.py```.
- [x] Ecrire une jolie docstring au début du fichier ```model.py```.
- [x] Ecrire une jolie docstring au début du fichier ```log.py```.
- [x] Ecrire une jolie docstring au début du fichier ```training.py```.

--------------------

## Liens utiles

- [LIRIS-ACCEDE](https://liris-accede.ec-lyon.fr/)
- [MediaEval](http://www.multimediaeval.org/)
- [Login ACCEDE](https://liris-accede.ec-lyon.fr/files/database-download/download.php)
- [Thèse Yohan Baveye](https://tel.archives-ouvertes.fr/tel-01272240/document)

--------------------





