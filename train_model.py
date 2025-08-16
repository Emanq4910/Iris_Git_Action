# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder

sns.set(style='white')

# Load Data
dataset = pd.read_csv('iris.csv')

# Rename columns to lowercase with underscores
dataset.rename(columns={
    'SepalLengthCm': 'sepal_length',
    'SepalWidthCm': 'sepal_width',
    'PetalLengthCm': 'petal_length',
    'PetalWidthCm': 'petal_width',
    'Species': 'target'
}, inplace=True)

# Drop 'Id' column (not needed for ML)
dataset.drop(columns=['Id'], inplace=True)

# Encode target (species) to integers
le = LabelEncoder()
dataset['target'] = le.fit_transform(dataset['target'])

# Feature Engineering
dataset['sepal_length_width_ratio'] = dataset['sepal_length'] / dataset['sepal_width']
dataset['petal_length_width_ratio'] = dataset['petal_length'] / dataset['petal_width']

# Select Features
dataset = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
                   'sepal_length_width_ratio', 'petal_length_width_ratio', 'target']]

# Train-test split
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=44)
X_train = train_data.drop('target', axis=1).values.astype('float32')
y_train = train_data['target'].values.astype('int32')
X_test = test_data.drop('target', axis=1).values.astype('float32')
y_test = test_data['target'].values.astype('int32')

# Logistic Regression
logreg = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=100, multi_class='multinomial')
logreg.fit(X_train, y_train)
predictions_lr = logreg.predict(X_test)

cm_lr = confusion_matrix(y_test, predictions_lr)
f1_lr = f1_score(y_test, predictions_lr, average='micro')
prec_lr = precision_score(y_test, predictions_lr, average='micro')
recall_lr = recall_score(y_test, predictions_lr, average='micro')

train_acc_lr = logreg.score(X_train, y_train) * 100
test_acc_lr = logreg.score(X_test, y_test) * 100

# Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
predictions_rf = rf_clf.predict(X_test)

f1_rf = f1_score(y_test, predictions_rf, average='micro')
prec_rf = precision_score(y_test, predictions_rf, average='micro')
recall_rf = recall_score(y_test, predictions_rf, average='micro')

train_acc_rf = rf_clf.score(X_train, y_train) * 100
test_acc_rf = rf_clf.score(X_test, y_test) * 100

# Confusion Matrix Plot
def plot_cm(cm, target_name, title="Confusion Matrix", cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_name))
    plt.xticks(tick_marks, target_name, rotation=45)
    plt.yticks(tick_marks, target_name)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else 'black')
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('ConfusionMatrix.png', dpi=120)
    plt.show()

target_name = le.classes_  # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
plot_cm(cm_lr, target_name, title="Logistic Regression Confusion Matrix", normalize=True)

# Feature importance
importances = rf_clf.feature_importances_
labels = dataset.drop('target', axis=1).columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns=['feature', 'importance'])
features = feature_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x='importance', y='feature', data=features)
ax.set_xlabel('Importance', fontsize=14)
ax.set_ylabel('Feature', fontsize=14)
ax.set_title('Random Forest Feature Importances', fontsize=12)
plt.tight_layout()
plt.savefig('FeatureImportance.png')
plt.close()

# Save scores to file
with open('scores.txt', "w") as score:
    score.write("Random Forest Train Accuracy: %2.1f%%\n" % train_acc_rf)
    score.write("Random Forest Test Accuracy: %2.1f%%\n" % test_acc_rf)
    score.write("F1 Score: %2.1f%%\n" % f1_rf)
    score.write("Recall Score: %2.1f%%\n" % recall_rf)
    score.write("Precision Score: %2.1f%%\n" % prec_rf)
    score.write("\n")
    score.write("Logistic Regression Train Accuracy: %2.1f%%\n" % train_acc_lr)
    score.write("Logistic Regression Test Accuracy: %2.1f%%\n" % test_acc_lr)
    score.write("F1 Score: %2.1f%%\n" % f1_lr)
    score.write("Recall Score: %2.1f%%\n" % recall_lr)
    score.write("Precision Score: %2.1f%%\n" % prec_lr)
