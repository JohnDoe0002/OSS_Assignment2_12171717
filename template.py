# PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/JohnDoe0002/OSS_Assignment2_12171717

# import package
import sys
import pandas as pd      
# import train_test_split for Splitting the given DataFrame
from sklearn.model_selection import train_test_split
# import DecisionTreeClassifier for training the decision tree model
from sklearn.tree import DecisionTreeClassifier
# import RandomForestClassifier for training the random forest model
from sklearn.ensemble import RandomForestClassifier
# import accuracy_score, precision_score, recall_score to evaluate the performances of the model
from sklearn.metrics import accuracy_score, precision_score, recall_score
# import SVC for training the Support Vector Machine model
from sklearn.svm import SVC
# import StandardScaler for training the pipeline consists of a standard scaler
from sklearn.preprocessing import StandardScaler

def load_dataset(dataset_path):
    # To-Do: Implement this function
    # read CSV for making the pandas DataFrame and return the DataFrame
    return pd.read_csv(dataset_path)

def dataset_stat(dataset_df):
    # To-Do: Implement this function
    # Drop the target label from columns (axis=1 is column) and column size is assign to n_feats.
    n_feats = dataset_df.drop(['target'], axis=1).columns.size
    # n_class0 is the number of class 0 data in Target label on DataFrame
    n_class0 = dataset_df['target'].value_counts().sort_index()[0]
    # n_class1 is the number of class 1 data in Target label on DataFrame
    n_class1 = dataset_df['target'].value_counts().sort_index()[1]
    return n_feats, n_class0, n_class1  # return statistical analysis results

def split_dataset(dataset_df, testset_size):
    # To-Do: Implement this function
    x = dataset_df.drop(['target'], axis=1).values  # independant features , target column drop
    y = dataset_df['target'].values  # dependant variable, y is target label value
    # Splitting the given DataFrame and return train data, test data, train label, and test label
    # X_train: training data, X_test: Test data, y_train: training target data, y_test: test target data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=testset_size)
    return X_train, X_test, y_train, y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
    # To-Do: Implement this function
    # Create the model of decision tree classifier
    tree_model = DecisionTreeClassifier()
    # Training the decision tree classifier from the training set (x_train and y_train)
    tree_model.fit(x_train, y_train)
    # Predict class or regression value for x_test
    y_pred = tree_model.predict(x_test)
    # Accuracy classification score from y_test and y_pred
    acc = accuracy_score(y_test, y_pred)
    # Compute the precision score from y_test and y_pred
    prec = precision_score(y_test, y_pred)
    # Compute the recall score from y_test and y_pred
    recall = recall_score(y_test, y_pred)
    return acc, prec, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
    # To-Do: Implement this function
    # Create the model of random forest classifier
    model = RandomForestClassifier()
    # Training the random forest classifier from the training set (x_train and y_train)
    model.fit(x_train, y_train)
    # Predict class or regression value for x_test
    y_pred = model.predict(x_test)
    # Accuracy classification score from y_test and y_pred
    acc = accuracy_score(y_test, y_pred)
    # Compute the precision score from y_test and y_pred
    prec = precision_score(y_test, y_pred)
    # Compute the recall score from y_test and y_pred
    recall = recall_score(y_test, y_pred)
    return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test):
    # To-Do: Implement this function
    #
    # Create StandardScaler object for the pipeline consists of a standard scaler and
    sc = StandardScaler()
    # Compute the mean and std of x_train data to be used for later scaling
    sc.fit(x_train)

    x_train_std = sc.transform(x_train) # Perform standardization of x_train
    x_test_std = sc.transform(x_test)   # Perform standardization of x_test
    # Create the model of Support Vector Classification
    svm_model = SVC()
    # Training the random forest classifier from the training set (x_train_std and y_train_std)
    svm_model.fit(x_train_std, y_train)
    # Predict class or regression value for x_test_std
    y_pred = svm_model.predict(x_test_std)  # 테스트
    # Accuracy classification score from y_test and y_pred
    acc = accuracy_score(y_test, y_pred)
    # Compute the precision score from y_test and y_pred
    prec = precision_score(y_test, y_pred)
    # Compute the recall score from y_test and y_pred
    recall = recall_score(y_test, y_pred)
    return acc, prec, recall

def print_performances(acc, prec, recall):
    # Do not modify this function!
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", recall)

if __name__ == '__main__':
    # Do not modify the main script!
    data_path = sys.argv[1]
    data_df = load_dataset(data_path)

    n_feats, n_class0, n_class1 = dataset_stat(data_df)
    print("Number of features: ", n_feats)
    print("Number of class 0 data entries: ", n_class0)
    print("Number of class 1 data entries: ", n_class1)

    print("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
    x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

    acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
    print("\nDecision Tree Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
    print("\nRandom Forest Performances")
    print_performances(acc, prec, recall)

    acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
    print("\nSVM Performances")
    print_performances(acc, prec, recall)
