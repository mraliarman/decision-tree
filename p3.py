# Step 0: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing

# Step 1: Read both datasets
def read_datasets():
    df = pd.read_csv("Telecust1.csv")
    df_null = pd.read_csv("Telecust1-Null.csv")
    return df, df_null

def encode_categorical_data(df):
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df
# Step 2: Handle missing values
def handle_missing_values(df):
    # Method 1: Delete lines with NaN data
    df_dropna = df.dropna()
    
    # Method 2: Replace with the average value of the column
    df_fillna = df.fillna(df.mean())
    
    # Report the difference in the number of data
    diff_rows = len(df) - len(df_fillna)
    return df_dropna, df_fillna, diff_rows

# Step 3: Obtain correlation coefficients
def correlation_analysis(df):
    correlation_matrix = df.corr()
    return correlation_matrix

# Step 4: Draw a graph for marital status, income, and tenure
def draw_graph(df, col1, col2, col3):
    plt.scatter(df[col1], df[col2], c=df[col3], cmap='viridis')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f"{col1} vs {col2} colored by {col3}")
    plt.show()

# Step 5: Normalize income using different methods
def normalize_income(df):
    # Method 1: Min-Max Scaling
    df['income_minmax'] = preprocessing.MinMaxScaler().fit_transform(df[['income']])
    
    # Method 2: Standardization
    df['income_standardized'] = preprocessing.StandardScaler().fit_transform(df[['income']])
    
    # Method 3: Robust Scaling
    df['income_robust'] = preprocessing.RobustScaler().fit_transform(df[['income']])
    
    # Draw graphs for comparison
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.title('Min-Max Scaling')
    plt.hist(df['income_minmax'])
    
    plt.subplot(132)
    plt.title('Standardization')
    plt.hist(df['income_standardized'])
    
    plt.subplot(133)
    plt.title('Robust Scaling')
    plt.hist(df['income_robust'])
    
    plt.show()

# Step 6: Change the Employee column using smoothing by bin means method
def bin_means_smoothing(df, col, num_categories):
    df[col] = pd.qcut(df[col], q=num_categories, labels=False, duplicates='drop')
    df[col] = df[col].astype(float)
    return df


# Step 7: Train DecisionTreeClassifier and report confusion matrix
def train_decision_tree(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return cm, report

# Step 8: Repeat step 7 for the data set without NaN values
def train_decision_tree_no_nan(df, target_col):
    df_no_nan, _, _ = handle_missing_values(df)
    return train_decision_tree(df_no_nan, target_col)

# Step 9: Repeat step 7 for the complete dataset
def train_decision_tree_complete(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return cm, report

# Step 10: Use feature reduction, feature extraction, normalization, discretization, etc.
# and train Decision Tree, report results after each training session
# (You may implement specific methods for each technique as needed)

# Main execution
df, df_null = read_datasets()
data_encoded = encode_categorical_data(df)
df_dropna, df_fillna, diff_rows = handle_missing_values(data_encoded)
correlation_matrix = correlation_analysis(df)
draw_graph(df, 'marital', 'income', 'tenure')
normalize_income(df)
df_smoothed = bin_means_smoothing(df, 'employ', 3)
cm, report = train_decision_tree(df, 'custcat')
cm_no_nan, report_no_nan = train_decision_tree_no_nan(df, 'custcat')
cm_complete, report_complete = train_decision_tree_complete(df, 'custcat')
