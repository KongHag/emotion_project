from os import listdir
from os.path import isfile, join
import json
import matplotlib.pyplot as plt
mypath='.'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

min_train = 200
min_test = 200
for filename in onlyfiles:
    if '.json' not in filename:
        continue
    with open(filename,'r') as file:
        a=json.load(file)
    print(a['num_hidden'], a['seq_len'],a['hidden_size'], min(a['test_losses']))
    name = filename.split('.')[0]
    min_train = min(min_train, a['train_losses'][-1])
    min_test = min(min_test, a['test_losses'][-1])
    plt.figure()
    plt.plot(a['train_losses'], label = 'train')
    plt.plot(a['test_losses'], label = 'test')
    plt.title(name)
    plt.savefig('graphs/' + name)
print('Minimum train loss after training : ' + str( min_train ) + '    ----------------     training : ' + name)
print('  ************     ')
print('Minimum test loss after training : ' + str( min_test ) + '    ----------------     training : ' + name)

