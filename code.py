import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Step 1: Read the dataset
def read_dataset(file_path):
    return pd.read_csv(file_path)

# Step 2: Encode categorical data using LabelEncoder
def encode_categorical_data(data):
    label_encoder = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data

# Step 3: Handle missing values by dropping or replacing with the mean
def handle_missing_values(data):
    # First method: Delete rows with NaN values
    data_dropped = data.dropna()
    
    # Second method: Replace NaN values with the mean of the column
    data_filled = data.fillna(data.mean())
    
    # Report the difference in the number of data
    difference = len(data) - len(data_dropped)
    print(f"Difference in the number of data after handling missing values: {difference}")
    
    return data_filled

# Step 4: Obtain and visualize correlation coefficients
def visualize_correlation(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(16, 14))
    
    # Use matshow instead of imshow to display the exact values
    plt.matshow(correlation_matrix, cmap='coolwarm', fignum=1)
    plt.colorbar(label='Correlation')
    
    # Display the values
    for (i, j), z in np.ndenumerate(correlation_matrix):
        plt.text(j, i, f'{z:.3f}', ha='center', va='center', fontsize=8)
    
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation='vertical')
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title("Correlation Matrix")
    # plt.savefig('correlation_matrix.png')  # Save the correlation matrix plot
    plt.show()

# Step 5: Draw a graph comparing marital status, income, and tenure
def compare_columns_graph(data):
    plt.scatter(data['marital'], data['income'])
    plt.title('Comparison of Marital Status and Income')
    plt.xlabel("Marital Status")
    plt.ylabel("income")
    plt.show()

    plt.scatter(data['marital'], data['tenure'])
    plt.title('Comparison of Marital Status and Tenure')
    plt.xlabel("Marital Status")
    plt.ylabel("Tenure")
    plt.show()

    plt.scatter(data['income'], data['tenure'])
    plt.title('Comparison of Marital Status and Tenure')
    plt.xlabel("income")
    plt.ylabel("Tenure")
    plt.show()

    plt.scatter(data['marital'], data['income'], c=data['tenure'], cmap='viridis', marker='o')
    plt.xlabel('Marital Status')
    plt.ylabel('Income')
    plt.colorbar(label='Tenure')
    plt.savefig('marital_income_tenure.png')  # Save the comparison graph
    plt.show()

# Step 6: Normalize two columns using different methods
def normalize_columns(data):
    # Normalize income using three different methods: Min-Max scaling, Z-score normalization, and Robust scaling   
    # Set a custom range for the histograms
    income_range = (0, 200)  # Adjust this range based on your data distribution
    
    data['income_minmax'] = (data['income'] - data['income'].min()) / (data['income'].max() - data['income'].min())
    data['income_zscore'] = (data['income'] - data['income'].mean()) / data['income'].std()
    data['income_robust'] = (data['income'] - data['income'].median()) / (data['income'].quantile(0.75) - data['income'].quantile(0.25))
    
    # Plot the graphs for comparison
    plt.figure(figsize=(15, 5))
    
    # Adjust the bin count and range for better visualization
    plt.subplot(1, 4, 1)
    plt.hist(data['income'], bins=50, range=income_range, color='blue', alpha=0.7)
    plt.title('Original Income Distribution')
    
    plt.subplot(1, 4, 2)
    plt.hist(data['income_minmax'], bins=50, range=(0, 1), color='green', alpha=0.7)
    plt.title('Min-Max Scaling')
    
    plt.subplot(1, 4, 3)
    plt.hist(data['income_zscore'], bins=50, range=(-3, 3), color='orange', alpha=0.7)
    plt.title('Z-score Normalization')
    
    plt.subplot(1, 4, 4)
    plt.hist(data['income_robust'], bins=50, range=(-3, 3), color='red', alpha=0.7)
    plt.title('Robust Scaling')
    
    plt.tight_layout()
    plt.savefig('income_normalization.png')  # Save the normalization comparison plot
    plt.show()

     # Set a custom range for the histograms
    tenure_range = (0, 200)  # Adjust this range based on your data distribution
    
    data['tenure_minmax'] = (data['tenure'] - data['tenure'].min()) / (data['tenure'].max() - data['income'].min())
    data['tenure_zscore'] = (data['tenure'] - data['tenure'].mean()) / data['tenure'].std()
    data['tenure_robust'] = (data['tenure'] - data['tenure'].median()) / (data['tenure'].quantile(0.75) - data['income'].quantile(0.25))
    
    # Plot the graphs for comparison
    plt.figure(figsize=(15, 5))
    
    # Adjust the bin count and range for better visualization
    plt.subplot(1, 4, 1)
    plt.hist(data['tenure'], bins=50, range=tenure_range, color='blue', alpha=0.7)
    plt.title('Original tenure Distribution')
    
    plt.subplot(1, 4, 2)
    plt.hist(data['tenure_minmax'], bins=50, range=(0, 1), color='green', alpha=0.7)
    plt.title('Min-Max Scaling')
    
    plt.subplot(1, 4, 3)
    plt.hist(data['tenure_zscore'], bins=50, range=(-3, 3), color='orange', alpha=0.7)
    plt.title('Z-score Normalization')
    
    plt.subplot(1, 4, 4)
    plt.hist(data['tenure_robust'], bins=50, range=(-3, 3), color='red', alpha=0.7)
    plt.title('Robust Scaling')
    
    plt.tight_layout()
    plt.savefig('income_normalization.png')  # Save the normalization comparison plot
    plt.show()


# Step 7: Smooth the 'employ' column using bin means method
def smooth_employ_column(data, num_categories):
    data['employ_smoothed'] = pd.cut(data['employ'], bins=num_categories, labels=False)
    # Report the average of the categories
    category_avg = data.groupby('employ_smoothed')['employ'].mean()
    print(f"Average of categories after smoothing 'employ' column:\n{category_avg}")

# Step 8: confusion matrix and criteria
def report_confusion_matrix_criteria(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    precision = cm[1, 1] / np.sum(cm[:, 1])
    recall = cm[1, 1] / np.sum(cm[1, :])
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

# Step 9: Split the dataset into training and testing sets
def split_train_test_data(data):
    X = data.drop('custcat', axis=1)
    y = data['custcat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


file_path = 'Telecust1.csv'
# file_path = 'Telecust1-Null.csv'
data = read_dataset(file_path)
data_encoded = encode_categorical_data(data)
data_filled = handle_missing_values(data_encoded)
visualize_correlation(data_filled)
compare_columns_graph(data_filled)
normalize_columns(data_filled)
smooth_employ_column(data_filled, num_categories=5)
X_train, X_test, y_train, y_test = split_train_test_data(data_filled)

# Step 10: Example usage of the decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.tree import plot_tree
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X_train.columns, class_names=list(map(str, model.classes_)), filled=True, rounded=True)
plt.show()

# Report confusion matrix and criteria
report_confusion_matrix_criteria(y_test, y_pred)

from sklearn.decomposition import PCA
def reduce_features(data, num_components=9):
    # Extract features (X) and target variable (y)
    X = data.drop('custcat', axis=1)
    y = data['custcat']

    # Apply PCA
    pca = PCA(n_components=num_components)
    X_reduced = pca.fit_transform(X)
    reduced_data = pd.DataFrame(data=X_reduced, columns=[f'PC{i}' for i in range(1, num_components + 1)])
    reduced_data['custcat'] = y

    return reduced_data

def train_decision_tree_with_parameters(X_train, y_train, criterion='gini'):
    model = DecisionTreeClassifier(criterion=criterion)
    model.fit(X_train, y_train)
    return model

reduced_data = reduce_features(data_filled)
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = split_train_test_data(reduced_data)

# Train decision tree with default parameters
model_default = train_decision_tree_with_parameters(X_train, y_train)
y_pred_default = model_default.predict(X_test)
print("Results for the default decision tree:")
report_confusion_matrix_criteria(y_test, y_pred_default)

# Train decision tree with different criterion
model_entropy = train_decision_tree_with_parameters(X_train, y_train, criterion='entropy')
y_pred_entropy = model_entropy.predict(X_test)
print("\nResults for the decision tree with entropy criterion:")
report_confusion_matrix_criteria(y_test, y_pred_entropy)

# Train decision tree on reduced features
model_reduced = train_decision_tree_with_parameters(X_train_reduced, y_train_reduced)
y_pred_reduced = model_reduced.predict(X_test_reduced)
print("\nResults for the decision tree on reduced features:")
report_confusion_matrix_criteria(y_test_reduced, y_pred_reduced)
