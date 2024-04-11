import pandas as pd
import matplotlib.pyplot as plt

features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE", "status"]
df = pd.read_csv('/content/data.csv', names=features)

df.describe()

df['status'].value_counts()

df.groupby('status').mean()

"""Plotting class imbalance"""

class_counts = df['status'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(class_counts.index, class_counts.values, color=['blue', 'red'])
plt.xlabel('Parkinson\'s Status')
plt.ylabel('Number of Patients')
plt.title('Class Distribution')
plt.xticks([0, 1], ['Healthy', 'Parkinson\'s'])
plt.show()

"""Showing Imbalance"""

yes =df[df['status']==1]
no = df[df['status']==0]
print(yes.shape,no.shape)

import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
import seaborn as sns
import math

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

n_features = X.shape[1]

n_rows = math.ceil(math.sqrt(n_features))
n_cols = math.ceil(n_features / n_rows)

print(X.shape)
print("Initial class distribution:", Counter(Y))

plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns[:-1]):
    plt.subplot(n_rows, n_cols, i+1)
    sns.kdeplot(data=df, x=col, hue='status', fill=True)
    plt.title(f'Feature: {col}')

plt.tight_layout()
plt.show()

"""Handling Imbalance"""

smote = SMOTE()
# : This line applies SMOTE to the feature array X and target variable Y. It generates synthetic samples for the minority class to match the number of samples in the majority class,
X_resampled, Y_resampled = smote.fit_resample(X, Y)

print("Class distribution after SMOTE:", Counter(Y_resampled))

X_resampled_df = pd.DataFrame(X_resampled, columns=df.columns[:-1])
X_resampled_df['status'] = Y_resampled

plt.figure(figsize=(12, 8))
for i, col in enumerate(X_resampled_df.columns[:-1]):
    plt.subplot(n_rows, n_cols, i+1)
    sns.kdeplot(data=X_resampled_df, x=col, hue='status', fill=True)
    plt.title(f'Feature: {col}')

plt.tight_layout()
plt.show()

df=X_resampled_df

"""Correlation matrix"""

corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
# This line creates a heatmap using seaborn's heatmap function. It visualizes the correlation matrix corr_matrix, with annotations (annot=True) displaying the correlation
# coefficients on each cell of the heatmap. The color map 'coolwarm' is used to represent different correlation values,
#  where cooler colors indicate negative correlation, warmer colors indicate positive correlation, and white represents no correlation.
# The fmt=".2f" parameter formats the annotations to two decimal places.
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

"""Outlier Detection"""

plt.figure(figsize=(10, 8))
#  The orient="h" parameter specifies that the boxplot should be horizontal. The palette="Set2" parameter sets the color palette for the plot.
sns.boxplot(data=df.drop(columns=['status']), orient="h", palette="Set2")
plt.title('Boxplot of Features')
plt.show()

"""Histogram"""

plt.figure(figsize=(24, 24))
for i, column in enumerate(df.columns[:-1]):
    plt.subplot(n_rows, n_cols, i+1)
    sns.histplot(df[column], kde=True, color='skyblue')
    plt.title(column)
plt.tight_layout()
plt.show()

"""Violin Plot"""

plt.figure(figsize=(24, 24))
for i, column in enumerate(df.columns[:-1]):
    plt.subplot(n_rows, n_cols, i+1)
    sns.violinplot(x='status', y=column, data=df, hue='status', legend=False)
    plt.title(column)
plt.tight_layout()
plt.show()

"""Scaling using Standard Scaler"""

from sklearn.preprocessing import StandardScaler
X = df.iloc[:, :-1] # all columns except the last one
Y = df.iloc[:, -1] # last column

# scale the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# convert X_scaled back to a DataFrame
df_scaled = pd.DataFrame(X_scaled, columns=df.columns[:-1])

"""After Scaling"""

plt.figure(figsize=(24, 24))
for i, column in enumerate(df_scaled.columns):
    plt.subplot(n_rows, n_cols, i+1)
    sns.histplot(df_scaled[column], kde=True, color='skyblue')
    plt.title(column)
plt.tight_layout()
plt.show()

"""Boxplot after scaling

"""

plt.figure(figsize=(10, 8))
sns.boxplot(df_scaled, orient="h", palette="Set2")
plt.title('Boxplot of Features')
plt.show()

"""Violin Plot after scaling"""

plt.figure(figsize=(24, 24))
a=df['status']
for i, column in enumerate(df_scaled.columns):
    plt.subplot(n_rows, n_cols, i+1)
    sns.violinplot(x=a, y=column, data=df_scaled, hue=a, legend=False)
    plt.title(column)
plt.tight_layout()
plt.show()

"""Splitting into training and test data"""

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
# 80% of the data used for training and 20% for testing.
X_train, X_test, Y_train, Y_test = train_test_split(df_scaled, Y, test_size=0.2, random_state=42)

print(X.shape,X_train.shape, X_test.shape)

"""Logistic Regression"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_curve, auc
import matplotlib.pyplot as plt

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred_logreg = logreg.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(Y_test, Y_pred_logreg))
print("Logistic Regression MCC:", matthews_corrcoef(Y_test, Y_pred_logreg))

Y_pred_proba_logreg = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba_logreg)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

"""Naive Baiyes"""

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, Y_train)

Y_pred_gnb = gnb.predict(X_test)

print("Naive Bayes Accuracy:", accuracy_score(Y_test, Y_pred_gnb))
print("Naive Bayes MCC:", matthews_corrcoef(Y_test, Y_pred_gnb))

Y_pred_proba_gnb = gnb.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba_gnb)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Gaussian Naive Bayes')
plt.legend(loc="lower right")
plt.show()

"""SVM"""

from sklearn.svm import SVC

svm = SVC(kernel='linear')

svm.fit(X_train, Y_train)

Y_pred_svm = svm.predict(X_test)

print("SVM Accuracy:", accuracy_score(Y_test, Y_pred_svm))
print("SVM MCC:", matthews_corrcoef(Y_test, Y_pred_svm))

Y_pred_proba_svm = svm.decision_function(X_test)
fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba_svm)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Support Vector Machine')
plt.legend(loc="lower right")
plt.show()

"""Neural Network"""

from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=200)

nn.fit(X_train, Y_train)

Y_pred_nn = nn.predict(X_test)

print("Neural Network Accuracy:", accuracy_score(Y_test, Y_pred_nn))
print("Neural Network MCC:", matthews_corrcoef(Y_test, Y_pred_nn))

Y_pred_proba_nn = nn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba_nn)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Neural Network')
plt.legend(loc="lower right")
plt.show()

"""Decision Tree"""

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)

Y_pred_dt = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(Y_test, Y_pred_dt))
print("Decision Tree MCC:", matthews_corrcoef(Y_test, Y_pred_dt))

Y_pred_proba_dt = dt.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba_dt)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Decision Tree')
plt.legend(loc="lower right")
plt.show()

"""Random Forrest"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rf = RandomForestClassifier(n_estimators=2)

rf.fit(X_train, Y_train)

Y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(Y_test, Y_pred_rf))
print("Random Forest MCC:", matthews_corrcoef(Y_test, Y_pred_rf))

Y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba_rf)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Random Forest')
plt.legend(loc="lower right")
plt.show()

"""Gradient Boosting"""

gb = GradientBoostingClassifier(n_estimators=2)

gb.fit(X_train, Y_train)

Y_pred_gb = gb.predict(X_test)

print("Gradient Boosting Accuracy:", accuracy_score(Y_test, Y_pred_gb))
print("Gradient Boosting MCC:", matthews_corrcoef(Y_test, Y_pred_gb))

Y_pred_proba_gb = gb.predict_proba(X_test)[:, 1]
fpr_gb, tpr_gb, _ = roc_curve(Y_test, Y_pred_proba_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

plt.figure()
plt.plot(fpr_gb, tpr_gb, color='green', lw=2, label='Gradient Boosting (area = %0.2f)' % roc_auc_gb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

"""K Neighbours"""

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=50)

knn.fit(X_train, Y_train)

Y_pred_knn = knn.predict(X_test)

print("KNN Accuracy:", accuracy_score(Y_test, Y_pred_knn))
print("KNN MCC:", matthews_corrcoef(Y_test, Y_pred_knn))

Y_pred_proba_knn = knn.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(Y_test, Y_pred_proba_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure()
plt.plot(fpr_knn, tpr_knn, color='purple', lw=2, label='KNN (area = %0.2f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - KNN')
plt.legend(loc="lower right")
plt.show()

"""Identifying top 10 features"""

gb = GradientBoostingClassifier(n_estimators=100)
gb.fit(X_train, Y_train)

# Get feature importances
feature_importances = gb.feature_importances_

# Create a DataFrame to store feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': df.columns[:-1], 'Importance': feature_importances})

# Sort the DataFrame by importance scores in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the top N most important features
top_n = 10  # Change this number to the desired number of top features
print(f"Top {top_n} most important features:")
print(feature_importance_df.head(top_n))

"""Hyperparameter Tuning"""

from sklearn.model_selection import GridSearchCV

models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.5]
    }
}

best_models = {}
for name, model in models.items():
    print(f"Tuning hyperparameters for {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best hyperparameters for {name}: {grid_search.best_params_}")

print("\nModel Performance Comparison:")
for name, model in best_models.items():
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"{name}: Accuracy = {accuracy:.4f}")

"""Final Output for 0"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Train the Gradient Boosting model (assuming it's the most accurate)
gb = GradientBoostingClassifier()
gb.fit(X_train, Y_train)

# Fit the scaler to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Function to predict Parkinson's disease based on input feature values
def predict_parkinsons(feature_values, scaler, model):
    # Scale the input feature values
    scaled_values = scaler.transform([feature_values])

    # Make a prediction using the trained model
    prediction = model.predict(scaled_values)

    if prediction[0] == 1:
        print("According to the model, you may have Parkinson's disease.")
    else:
        print("According to the model, you do not have Parkinson's disease.")

# Example usage
feature_values = [0.671187,	0.290739,	0.665351,	-0.486884,	-0.592643	,-0.466668,	-0.505222	,-0.466631,	-0.643002	,-0.613720,	-0.608330	,-0.614510,	-0.637553	,-0.608315	,-0.331501	,0.632651,	-0.539993	,-0.405544	,-0.988466,	-0.795974,	-0.595405,	-0.929319,0]
# Modify the feature_values list to remove the extra feature
feature_values = feature_values[:-1]

# Call the predict_parkinsons function with the modified feature_values and fitted scaler
predict_parkinsons(feature_values, scaler, gb)

"""Final Output for 1"""

feature_values = [-0.219163, -0.094935, -0.217257, 0.158983, 0.193516, 0.152381, 0.164970, 0.152369, 0.209960, 0.200398, 0.198638, 0.200656, 0.208180, 0.198633, 0.108245, -0.206580, 0.176324, 0.132422, 0.322765, 0.259910, 0.194418, 0.303451]

predict_parkinsons(feature_values, scaler, gb)

df_concatenated = pd.concat([df_scaled, Y], axis=1)
pd.set_option('display.max_columns', None)
df_concatenated
df_concatenated.groupby('status').mean()

"""Calculating the Accuracy, Precision, Recall and F1 Score for all the models"""

from sklearn.metrics import precision_score, recall_score, f1_score

#function to calculate precision, recall, and F1 score
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

# Calculate metrics for each model
models = {
    "Logistic Regression": Y_pred_logreg,
    "Naive Bayes": Y_pred_gnb,
    "SVM": Y_pred_svm,
    "Neural Network": Y_pred_nn,
    "Decision Tree": Y_pred_dt,
    "Random Forest": Y_pred_rf,
    "Gradient Boosting": Y_pred_gb,
    "KNN": Y_pred_knn
}

print("Metrics for each model:")
for model_name, y_pred in models.items():
    precision, recall, f1 = calculate_metrics(Y_test, y_pred)
    print(f"{model_name}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
