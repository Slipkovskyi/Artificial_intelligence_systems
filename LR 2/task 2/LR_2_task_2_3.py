import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
input_file = 'income_data.txt'
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

# Read data
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line.strip().split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(0)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)
            count_class2 += 1

# Convert to numpy array
X = np.array(X)

# Identify numeric and categorical columns
numeric_features = []
categorical_features = []

for i, item in enumerate(X[0]):
    try:
        float(item)
        numeric_features.append(i)
    except ValueError:
        categorical_features.append(i)

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', preprocessing.StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a full pipeline that includes preprocessing
classifier_sigmoid = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='sigmoid'))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Train the model using the pipeline (which handles preprocessing)
classifier_sigmoid.fit(X_train, y_train)

# Prediction and evaluation
y_pred_sigmoid = classifier_sigmoid.predict(X_test)

print("Sigmoid Kernel:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_sigmoid) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred_sigmoid, average='weighted') * 100:.2f}%")
print(f"Recall: {recall_score(y_test, y_pred_sigmoid, average='weighted') * 100:.2f}%")
print(f"F1 Score: {f1_score(y_test, y_pred_sigmoid, average='weighted') * 100:.2f}%")