import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Function to draw and save the graph
def draw_and_save_graph(x, y, xlabel, ylabel, title, filename):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.show()

# Read datasets
Telecust1_path = 'Telecust1.csv'  # Replace with the actual path
test_data_path = 'Telecust1-Null.csv'  # Replace with the actual path
main_df = pd.read_csv(Telecust1_path)
test_df = pd.read_csv(test_data_path)

# Convert string data to numbers using LabelEncoder
label_encoder = LabelEncoder()
for col in main_df.columns:
    if main_df[col].dtype == 'object':
        main_df[col] = label_encoder.fit_transform(main_df[col])

# Method 1: Delete lines containing NaN data
main_df_method1 = main_df.dropna()
diff_method1 = len(main_df) - len(main_df_method1)

# Method 2: Replace with the average value of the column
main_df_method2 = main_df.fillna(main_df.mean())
diff_method2 = len(main_df) - len(main_df_method2)

# Use the data set obtained from the second method for the next steps
main_df = main_df_method2

# Obtain correlation coefficients
correlation_matrix = main_df.corr()

# Draw the relationship between marital status, income, and tenure
draw_and_save_graph(main_df['marital'], main_df['income'], 'Marital Status', 'Income', 'Marital Status vs Income', 'marital_vs_income.png')
draw_and_save_graph(main_df['marital'], main_df['tenure'], 'Marital Status', 'Tenure', 'Marital Status vs Tenure', 'marital_vs_tenure.png')

# Normalize the columns 'income' and 'tenure' using different methods
# (Assuming you meant 'tenure' instead of 'with' in the question)
main_df['income_min_max'] = (main_df['income'] - main_df['income'].min()) / (main_df['income'].max() - main_df['income'].min())
main_df['income_standard'] = (main_df['income'] - main_df['income'].mean()) / main_df['income'].std()

# Change the 'employ' column using the smoothing by bin means method
desired_categories = 5  # Replace with your desired number of categories
main_df['employ_smoothed'] = main_df.groupby('employ')['income'].transform('mean')

# Report the average of the categories
average_categories = main_df.groupby('employ')['employ_smoothed'].mean()

# Research the confusion matrix and the criteria
X_train, X_test = train_test_split(main_df, test_size=0.2, random_state=42)
y_true = X_test['custcat']
y_pred = X_test['custcat']  # Replace with your prediction logic

conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / len(X_test)
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

# Print results
def print_results():
    print(f"Method 1: Deleted {diff_method1} records with NaN values.")
    print(f"Method 2: Replaced NaN values with the average. Deleted {diff_method2} records.")
    print("\nCorrelation Coefficients:")
    print(correlation_matrix)
    print("\nAverage Categories:")
    print(average_categories)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

# Call the print_results function
print_results()
