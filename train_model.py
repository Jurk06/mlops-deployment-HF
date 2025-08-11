# train_model_classification.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

sns.set(style='white')

# Load Data
dataset = pd.read_csv('iris.csv')

# Expect columns like: "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "target"
# Clean column names similar to your original code
dataset.columns = [colname.strip(' (cm)').replace(" ", "_") for colname in dataset.columns.tolist()]

# Feature Engineering
dataset['sepal_length_width_ratio'] = dataset['sepal_length'] / dataset['sepal_width']
dataset['petal_length_width_ratio'] = dataset['petal_length'] / dataset['petal_width']

# Keep only features + target
dataset = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
                   'sepal_length_width_ratio', 'petal_length_width_ratio', 'target']]

# Ensure target is categorical (integers or strings are fine for sklearn)
# If the CSV already has 0/1/2 labels or species strings, this will work as-is.
# If target is numeric floats, cast to int.
if np.issubdtype(dataset['target'].dtype, np.floating):
    dataset['target'] = dataset['target'].astype(int)

# Train-test split
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=44, stratify=dataset['target'])
X_train = train_data.drop('target', axis=1).values.astype('float32')
y_train = train_data['target'].values
X_test = test_data.drop('target', axis=1).values.astype('float32')
y_test = test_data['target'].values

# Random Forest classifier
clf = RandomForestClassifier(random_state=44, n_estimators=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

# Metrics
train_acc = clf.score(X_train, y_train) * 100
test_acc = accuracy_score(y_test, y_pred) * 100
report = classification_report(y_test, y_pred, digits=4)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
classes = np.unique(y_train)

# Plot feature importances
importances = clf.feature_importances_
labels = dataset.drop('target', axis=1).columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns=['feature', 'importance'])
features = feature_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x='importance', y='feature', data=features, color='steelblue')
ax.set_title('Random Forest Feature Importances (Classification)', fontsize=14)
plt.tight_layout()
plt.savefig('FeatureImportance.png')
plt.close()

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('ConfusionMatrix.png')
plt.close()

# Save metrics
with open('scores.txt', "w") as score:
    score.write("Random Forest Train Accuracy: %2.1f%%\n" % train_acc)
    score.write("Random Forest Test Accuracy: %2.1f%%\n" % test_acc)
    score.write("\nClassification Report:\n")
    score.write(report)

# Save the model
joblib.dump(clf, "model.pkl")

print("✅ Classification training complete. Metrics saved to scores.txt, plots saved, model saved to model.pkl")
