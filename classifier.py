print('Importing libraries ...')
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import utilities as utils
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

print('Reading and cleaning data ...')

path = os.path.join('Asn2DataSet','TrainingDataset')
alphas = [utils.read_sequence(os.path.join(path, 'Alpha', seq)) for seq in os.listdir(os.path.join(path,'Alpha'))]
alphas = utils.clean_fasta(alphas)
betas = [utils.read_sequence(os.path.join(path, 'Beta', seq)) for seq in os.listdir(os.path.join(path,'Beta'))]
betas = utils.clean_fasta(betas)
deltas = [utils.read_sequence(os.path.join(path, 'Delta', seq)) for seq in os.listdir(os.path.join(path,'Delta'))]
deltas = utils.clean_fasta(deltas)
gammas = [utils.read_sequence(os.path.join(path, 'Gamma', seq)) for seq in os.listdir(os.path.join(path,'Gamma'))]
gammas = utils.clean_fasta(gammas)


print('Generating CGRs ...')

cgrs = []
labels = []
for seq in alphas:
    tmp = utils.cgr(seq, 'ACGT', 7)
    cgrs.append(tmp)
    labels.append('Alpha')

for seq in betas:
    tmp = utils.cgr(seq, 'ACGT', 7)
    cgrs.append(tmp)
    labels.append('Beta')

for seq in deltas:
    tmp = utils.cgr(seq, 'ACGT', 7)
    cgrs.append(tmp)
    labels.append('Delta')

for seq in gammas:
    tmp = utils.cgr(seq, 'ACGT', 7)
    cgrs.append(tmp)
    labels.append('Gamma')

cgrs = np.array(cgrs)
labels = np.array(labels)


print('Normalizing feature vectors ...')
cgr_vectors = np.array([mat.flatten() for mat in cgrs])
normalized_cgr_vectors = np.array([vector/np.max(vector) for vector in cgr_vectors])


label_map = {label: i for i,label in enumerate(np.unique(labels))}
dummy_labels = np.array([label_map[lab] for lab in labels])
numerical_to_label = {val: key for key,val in label_map.items()}


variants_dict = {variant: np.sum(cgrs[np.where(labels== variant)], axis=0) for variant in np.unique(labels)}


print('Plotting CGRs ...')

fig,ax = plt.subplots(1,4, figsize=(10,3))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
fig.suptitle('Chaos Game Representation - CGR')
for i,variant in enumerate(variants_dict):
    ax[i].imshow(variants_dict[variant], cmap='gray_r')
    ax[i].set_title(variant)
    ax[i].set_yticks([])
    ax[i].set_xticks([])
plt.show()

print('Initializing Random Forest Classifier ...')

clf_rf = RandomForestClassifier(criterion='gini',max_depth=3)

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=99)

print('Evaluating the model using K-Folds ...')

accs_rf = []
for train_i, test_j in kf.split(normalized_cgr_vectors):
    X_train, X_test = normalized_cgr_vectors[train_i], normalized_cgr_vectors[test_j]
    y_train, y_test = dummy_labels[train_i], dummy_labels[test_j]
    clf_rf.fit(X_train, y_train)
    y_preds = clf_rf.predict(X_test)
    accuracy = 100*accuracy_score(y_test, y_preds)
    accs_rf.append(accuracy)

df = pd.DataFrame({'Fold': range(1, FOLDS + 1),'Accuracy': accs_rf})
df.set_index('Fold', inplace=True)
print(df)

print('Training the model on the dataset ...')

clf_rf.fit(normalized_cgr_vectors, dummy_labels)

print('Preprocessing testing data ...')
test_seqs = []
paths = os.path.join('Asn2DataSet', 'TestingDataset')
testing_path = os.listdir(os.path.join('Asn2DataSet','TestingDataset'))
for fasta in testing_path:
    test_seqs.append(utils.read_sequence(os.path.join(paths, fasta)))
test_seqs = utils.clean_fasta(test_seqs)
test_cgrs = [utils.cgr(seq, 'ACGT', 7) for seq in test_seqs]
cgr_vectors_test = np.array([mat.flatten() for mat in test_cgrs])
normalized_cgr_vectors_test = np.array([vector/np.max(vector) for vector in cgr_vectors_test])

print('Predicting testing data labels ...')

preds = clf_rf.predict(normalized_cgr_vectors_test)

df1 = pd.DataFrame({'TestData': [f'{i}.fasta' for i in range(1, 11)], 'PredictedLabel': [numerical_to_label[pred] for pred in preds]})
df1.set_index('TestData', inplace=True)

print(df1)