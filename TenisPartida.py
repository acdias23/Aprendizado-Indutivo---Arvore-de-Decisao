#pip install 
#pip install matplotlib
#pip install pandas

import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff


data,meta = arff.loadarff('./TenisPartida.arff')

attributes = meta.names()
data_value = np.asarray(data)


Jogador1 = np.asarray(data['Jogador1']).reshape(-1,1)
Jogador2 = np.asarray(data['Jogador2']).reshape(-1,1)
Faltas = np.asarray(data['Faltas']).reshape(-1,1)
features = np.concatenate((Jogador1 , Jogador2, Faltas),axis=1)
target = data['resultado']


Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

plt.figure(figsize=(10, 6.5))
tree.plot_tree(Arvore,feature_names=['Jogador1','Jogador2', 'Faltas'],class_names=['Vitoria', 'Derrota'],
                   filled=True, rounded=True)
plt.show()

fig, ax = plt.subplots(figsize=(25, 10))
metrics.ConfusionMatrixDisplay.from_estimator(Arvore,features,target,display_labels=['Vitoria', 'Derrota'], values_format='d', ax=ax)
plt.show()