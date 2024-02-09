import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Function to read the dataset
def read_dataset(file_path):
    return pd.read_csv(file_path)

# Function to encode categorical data using LabelEncoder
def encode_categorical_data(df):
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df

# Function to handle missing values by dropping or replacing with the mean
def handle_missing_values(df):
    # First method: Delete rows with NaN values
    df_dropped = df.dropna()
    
    # Second method: Replace NaN values with the mean of the column
    df_filled = df.fillna(df.mean())
    
    # Report the difference in the number of data
    difference = len(df) - len(df_filled)
    print(f"Difference in the number of data after handling missing values: {difference}")
    
    return df_filled

# Function to obtain and visualize correlation coefficients
def visualize_correlation(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation='vertical')
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title("Correlation Matrix")
    plt.show()

# Function to draw a graph comparing marital status, income, and tenure
def compare_columns_graph(df):
    plt.scatter(df['marital'], df['income'], c=df['tenure'], cmap='viridis', marker='o')
    plt.xlabel('Marital Status')
    plt.ylabel('Income')
    plt.title('Comparison of Marital Status, Income, and Tenure')
    plt.colorbar(label='Tenure')
    plt.show()

# Function to normalize two columns using different methods
def normalize_columns(df):
    # Normalize income using three different methods: Min-Max scaling, Z-score normalization, and Robust scaling
    df['income_minmax'] = (df['income'] - df['income'].min()) / (df['income'].max() - df['income'].min())
    df['income_zscore'] = (df['income'] - df['income'].mean()) / df['income'].std()
    df['income_robust'] = (df['income'] - df['income'].median()) / (df['income'].quantile(0.75) - df['income'].quantile(0.25))
    
    # Plot the graphs for comparison
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.hist(df['income'], bins=20, color='blue', alpha=0.7)
    plt.title('Original Income Distribution')
    
    plt.subplot(1, 4, 2)
    plt.hist(df['income_minmax'], bins=20, color='green', alpha=0.7)
    plt.title('Min-Max Scaling')
    
    plt.subplot(1, 4, 3)
    plt.hist(df['income_zscore'], bins=20, color='orange', alpha=0.7)
    plt.title('Z-score Normalization')
    
    plt.subplot(1, 4, 4)
    plt.hist(df['income_robust'], bins=20, color='red', alpha=0.7)
    plt.title('Robust Scaling')
    
    plt.tight_layout()
    plt.show()

# Function to smooth the 'employ' column using bin means method
def smooth_employ_column(df, num_categories):
    df['employ_smoothed'] = pd.cut(df['employ'], bins=num_categories, labels=False)
    # Report the average of the categories
    category_avg = df.groupby('employ_smoothed')['employ'].mean()
    print(f"Average of categories after smoothing 'employ' column:\n{category_avg}")

# Function to research and report confusion matrix and criteria
def report_confusion_matrix_criteria():
    # Provide details about confusion matrix and criteria here
    pass

# Function to split the dataset into training and testing sets
def split_train_test_data(df):
    X = df.drop('custcat', axis=1)
    y = df['custcat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Example usage:
file_path = 'Telecust1.csv'
data = read_dataset(file_path)
data_encoded = encode_categorical_data(data)
data_filled = handle_missing_values(data_encoded)
visualize_correlation(data_filled)
compare_columns_graph(data_filled)
normalize_columns(data_filled)
smooth_employ_column(data_filled, num_categories=5)
report_confusion_matrix_criteria()
X_train, X_test, y_train, y_test = split_train_test_data(data_filled)
