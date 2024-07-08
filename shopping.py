import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

TEST_SIZE = 0.4

def main():


    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(r"C:\Users\pc\Desktop\shopping\shopping\shopping.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE, random_state=42
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence arrays and a list of labels. Return a tuple (evidence, labels).

    Columns:
    - Administrative: integer
    - Administrative_Duration: floating point number
    - Informational: integer
    - Informational_Duration: floating point number
    - ProductRelated: integer
    - ProductRelated_Duration: floating point number
    - BounceRates: floating point number
    - ExitRates: floating point number
    - PageValues: floating point number
    - SpecialDay: floating point number
    - Month: index from 0 (January) to 11 (December)
    - OperatingSystems: integer
    - Browser: integer
    - Region: integer
    - TrafficType: integer
    - VisitorType: integer (0 for not returning, 1 for returning)
    - Weekend: integer (0 for false, 1 for true)
    - Revenue: integer (1 if true, 0 otherwise)
    """
    # Load data into a pandas DataFrame
    # Define column names based on the provided description
    column_names = [
        'Administrative', 'Administrative_Duration',
        'Informational', 'Informational_Duration',
        'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates',
        'PageValues', 'SpecialDay',
        'Month', 'OperatingSystems',
        'Browser', 'Region',
        'TrafficType', 'VisitorType',
        'Weekend', 'Revenue'
    ]

    # Read CSV file into a DataFrame
    evidence = []
    labels = []

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row

        for row in csv_reader:
            # Extract features from the row
            features = [
                int(row[0]), float(row[1]),
                int(row[2]), float(row[3]),
                int(row[4]), float(row[5]),
                float(row[6]), float(row[7]),
                float(row[8]), float(row[9]),
                row[10], int(row[11]),
                int(row[12]), int(row[13]),
                int(row[14]), int(row[15]),
                int(row[16]), int(row[17])
            ]

            # Convert 'Month' to numeric index
            month_mapping = {
                'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3,
                'May': 4, 'Jun': 5, 'Jul': 6, 'Aug': 7,
                'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
            }
            features[10] = month_mapping[features[10]]

            # Convert 'VisitorType' and 'Weekend' to integers
            features[15] = 1 if features[15] == 'Returning_Visitor' else 0
            features[16] = 1 if features[16] == 'TRUE' else 0

            # Append features and label to lists
            evidence.append(features[:-1])  # Append all except the last column ('Revenue')
            labels.append(int(features[-1]))  # Append the last column ('Revenue') as label

    return evidence, labels


def train_model(evidence, labels):
    """
    Given evidence (features) and labels, train a k-nearest neighbor model (k=1).
    Return the trained model.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Given actual labels and predicted labels, calculate sensitivity (true positive rate)
    and specificity (true negative rate) and return them as a tuple (sensitivity, specificity).
    """
    # Calculate true positive rate (sensitivity)
    TP = sum((labels == 1) & (predictions == 1))
    FN = sum((labels == 1) & (predictions == 0))
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0.0

    # Calculate true negative rate (specificity)
    TN = sum((labels == 0) & (predictions == 0))
    FP = sum((labels == 0) & (predictions == 1))
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0.0

    return sensitivity, specificity

if __name__ == "__main__":
    main()
