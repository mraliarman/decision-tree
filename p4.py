# Step 0: Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

# Step 1: Read both datasets and encode categorical data
def encode_categorical_data(df):
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df

def read_and_encode_data(file_path):
    df = pd.read_csv(file_path)
    df = encode_categorical_data(df)
    return df

# Step 2: Handle missing values
def handle_missing_values(df):
    # Method 1: Delete lines containing NaN data
    df_method1 = df.dropna()

    # Method 2: Replace with the average value of the column
    df_method2 = df.fillna(df.mean())

    # Report the difference in the number of data
    diff_rows = len(df_method1) - len(df_method2)
    print(f"Method 1 rows: {len(df_method1)}, Method 2 rows: {len(df_method2)}, Difference: {diff_rows}")

    return df_method2

# Step 3: Obtain correlation coefficients and draw a graph
def correlation_and_graph(df):
    correlation_matrix = df.corr()
    plt.scatter(df['marital'], df['income'])
    plt.scatter(df['marital'], df['tenure'])
    plt.title("Scatter Plot of Marital Status, Income, and Tenure")
    plt.xlabel("Marital Status")
    plt.ylabel("Value")
    plt.show()
    return correlation_matrix

# Step 4: Change the Employee column using the smoothing by bin means method
def smooth_employee_column(df, num_categories):
    df['employ'] = pd.cut(df['employ'], bins=num_categories, labels=False)
    avg_by_category = df.groupby('employ')['employ'].mean()
    df['employ'] = df['employ'].apply(lambda x: avg_by_category[x])
    return df

# Step 5: Research confusion matrix and report criteria
def confusion_matrix_and_criteria(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    precision = cm[1, 1] / np.sum(cm[:, 1])
    recall = cm[1, 1] / np.sum(cm[1, :])
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

# Step 6: Split data into training and testing sets
def split_data(df):
    X = df.drop('custcat', axis=1)
    y = df['custcat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 7: Train decision tree model and analyze confusion matrix
def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion_matrix_and_criteria(y_test, y_pred)

# Step 8: Train decision tree model with data set where NaN data is removed
def train_decision_tree_no_nan(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion_matrix_and_criteria(y_test, y_pred)

# Step 9: Train decision tree model with the complete Telecust1.csv dataset
def train_decision_tree_full_data(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion_matrix_and_criteria(y_test, y_pred)

# Step 10: Use feature reduction, feature extraction, normalization, discretization, etc.
def train_decision_tree_with_preprocessing(X_train, X_test, y_train, y_test, scaler):
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = DecisionTreeClassifier()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    confusion_matrix_and_criteria(y_test, y_pred)

# Main execution
file_path = "Telecust1.csv"
df = read_and_encode_data(file_path)

# Step 1: Handle missing values
df_method2 = handle_missing_values(df)

# Step 2: Obtain correlation coefficients and draw a graph
correlation_matrix = correlation_and_graph(df_method2)

# Step 3: Normalize the two columns of income
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()
scaler_robust = RobustScaler()

# Draw graphs for different normalization methods
for scaler, label in zip([scaler_standard, scaler_minmax, scaler_robust], ['Standard Scaler', 'MinMax Scaler', 'Robust Scaler']):
    df_normalized = df_method2.copy()
    df_normalized[['income', 'tenure']] = scaler.fit_transform(df_normalized[['income', 'tenure']])
    plt.scatter(df_normalized['income'], df_normalized['tenure'])
    plt.title(f"Scatter Plot after {label} Normalization")
    plt.xlabel("Income")
    plt.ylabel("Tenure")
    plt.show()

# Step 4: Change the Employee column using the smoothing by bin means method
num_categories = 5
df_smoothed = smooth_employee_column(df_method2, num_categories)
average_categories = df_smoothed['employ'].mean()
print(f"Average of Categories after Smoothing: {average_categories}")

# Step 5: Research the confusion matrix and criteria
X_train, X_test, y_train, y_test = split_data(df_smoothed)
train_decision_tree(X_train, X_test, y_train, y_test)

# Step 6: Train decision tree model and analyze confusion matrix
X_train, X_test, y_train, y_test = split_data(df_smoothed)
train_decision_tree(X_train, X_test, y_train, y_test)

# Step 7: Train decision tree model with data set where NaN data is removed
df_no_nan = df.dropna()
X_train, X_test, y_train, y_test = split_data(df_no_nan)
train_decision_tree_no_nan(X_train, X_test, y_train, y_test)

# Step 8: Train decision tree model with the complete Telecust1.csv dataset
X_train, X_test, y_train, y_test = split_data(df)
train_decision_tree_full_data(X_train, X_test, y_train, y_test)

# Step 9: Use feature reduction, feature extraction, normalization, discretization, etc.
X_train, X_test, y_train, y_test = split_data(df)
train_decision_tree_with_preprocessing(X_train, X_test, y_train, y_test, scaler_standard)
train_decision_tree_with_preprocessing(X_train, X_test, y_train, y_test, scaler_minmax)
train_decision_tree_with_preprocessing(X_train, X_test, y_train, y_test, scaler_robust)
