import numpy as np
import pandas
import datetime
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import math

"""
Parameter descriptions:
model: sklearn model, refer to http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
data: training data set
test: testing pandas data frame, use EDA.loadData("dataFolder", "test")
predictors: an array of column names used to train the model ["x", "y"] (at least)
label: and array of column names we are trying to predict, "place_id"
predict: boolean to indicate whether or not we want to apply our model to the test set
output: base name for output file, try to use the same name as the training set used
modelDescription: an array of string values to describe the model used, e.g. ["knn", "k=25", "predictors=[x,y]"]

What this will do:
Trains a model depending on the parameters specified
If predict=True, apply the model to the test set, saving the predicted place_ids to output file
Accuracy is percentage of values the model correctly predicted
Cross-Validation is mean error of a 5-fold cross-validation process
"""
def classification_model(model, data, predictors, label):
    X = data[predictors] # Features
    y = data[label] # Target
    """Sample Usage
    Fit the model:
    model.fit(X, y)
    Make predictions on training set (not advised):
    predictions = model.predict(X)
    Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[label])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    """
    # Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits=5)
    error = []
    for train, test in kf.split(X):
        # Filter training data
        train_predictors = (X.iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = y.iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(X.iloc[test, :], y.iloc[test]))

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    # Fit the model again so that it can be referred outside the function:
    # model.fit(data[predictors], data[label])

def featureImportance(data):
    # HourMinute is sin(15*(Hour + Minute/60))
    # sample = data.sample(n=40000, replace=True, random_state=10)
    sample = data
    predictors = ['x', 'y', 'accuracy', 'HourMinute']
    X = sample[predictors]
    y = sample['place_id']
    # Make hour/minute sinusoidal --> hour 0 is "close" to hour 24
    forest = ExtraTreesClassifier(n_estimators=10, n_jobs=-1, random_state=0)
    print("Fitting the model...")
    forest.fit(X, y)
    print("Done...")
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature Ranking")

    for f in range(X.shape[1]):
        print('%d. %s (%f)' % (f + 1, predictors[indices[f]], importances[indices[f]]))

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    # plt.savefig('Figures/FeatureImportances3.png')
    # plt.show()



def main():
    # model = KNeighborsClassifier(n_jobs=-1, n_neighbors=25, p=1, weights="distance")
    model = ExtraTreesClassifier(n_estimators=10, n_jobs=-1, random_state=0)
    training = pandas.read_pickle("Data/train_46.pkl")
    predictors = ['x','y','accuracy','HourMinute']
    label = "place_id"
    classification_model(model, training, predictors, label)
    # 40.110% Cross-Validation Score KNN
    # 42.055$ Cross-Validation Score ETC35
    # 39.858% Cross-Validation Score ETC46(0.25)
    # 42.364% Cross-Validation Score ETC46(0.5)
    featureImportance(training)
if __name__=="__main__":
    main()

