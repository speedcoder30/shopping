import csv
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
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
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    df=pd.read_csv(filename,
                   dtype={"Administrative_Duration": float,
                          "Informational_Duration":float,
                          "ProductRelated_Duration":float,
                          "BounceRates":float,
                          "ExitRates":float,
                          "PageValues":float,
                          "SpecialDay":float,
                          "Administrative":int,
                          "Informational":int,
                          "ProductRelated":int,
                          "OperatingSystems":int,
                          "Browser":int,
                          "Region":int,
                          "TrafficType":int,
                          "Weekend":int,
                          "Revenue":int},
                   converters={'VisitorType':convert_vistorType,
                               'Month':convert_month_int}
                   )
    evidence=df.loc[:, df.columns != 'Revenue']
    labels=df["Revenue"]
    print("evidence",evidence)
    print("labels",labels)
    return (evidence.values.tolist(),labels.values.tolist())

def convert_vistorType(vistorType:str):
    if vistorType.lower()=="Returning_Visitor".lower():
        return 1
    else:
        return 0

def convert_month_int(month:str):
    months = dict(jan=0, feb=1, mar=2,apr=3,may=4,june=5,jul=6,aug=7,sep=8,oct=9,nov=10,dec=11)

    return months[month.lower()]

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence,labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    actual_postive_count=labels.count(1)
    actual_neagtive_count=labels.count(0)
    print(type(labels))
    print(predictions)
    postive_index_list=[i for i in range(len(labels)) if labels[i] == 1]
    negative_index_list=[i for i in range(len(labels)) if labels[i] == 0]

    successful_postive_count=0
    successful_negative_count=0
    for index in postive_index_list:
        if predictions[index]==1:
            successful_postive_count+=1

    for index in negative_index_list:
        if predictions[index]==0:
            successful_negative_count+=1

    return ((successful_postive_count/actual_postive_count),(successful_negative_count/actual_neagtive_count))

if __name__ == "__main__":
    main()
