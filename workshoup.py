from six.moves import urllib
from scipy.io import loadmat
import matplotlib
import numpy as np
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

sgd_clf = SGDClassifier(max_iter=5, random_state=42)






mnist= {}

def get_data():

    mnist_raw = loadmat('./mnist-original.mat')
    print(mnist_raw["data"].T[0])
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    X, y = mnist["data"], mnist["target"]

    some_digit = X[36000]
    some_digit
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    # plt.show()

    return X,y


def plot_digits(instances, images_per_row=70000, **options):
    size = 28
    print("len here",len(instances))

    images_per_row = min(len(instances), images_per_row)


    print("damn here",images_per_row)


    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")


X,y = get_data()
plot_digits(X)


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


print("nananan here",y_train)
print("nananan here",y_train==5)

y_train_5 = (y_train != 5)
y_test_5 = (y_test != 5)

shuffle_index = np.random.permutation(60000)
some_digit = X[36000]


# clf = make_pipeline(StandardScaler(),
#                     SGDClassifier(max_iter=5, tol=1e-3))
# clf.fit(X, y)
#
#
# print(clf.predict([some_digit]))

skfolds = StratifiedKFold(n_splits=3, random_state=None)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))



print("hoho here",cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# class Never5Classifier(BaseEstimator):
#     def fit(self, X, y=None):
#         skfolds = StratifiedKFold(n_splits=3, random_state=42)
#
#
#         for train_index, test_index in skfolds.split(X_train, y_train_5):
#             clone_clf = clone(sgd_clf)
#             X_train_folds = X_train[train_index]
#             y_train_folds = (y_train_5[train_index])
#             X_test_fold = X_train[test_index]
#             y_test_fold = (y_train_5[test_index])
#
#             clone_clf.fit(X_train_folds, y_train_folds)
#             y_pred = clone_clf.predict(X_test_fold)
#             n_correct = sum(y_pred == y_test_fold)
#             print(n_correct / len(y_pred))
#
#
#     def predict(self, X):
#         cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
#


# never_5_clf = Never5Classifier()
# print(never_5_clf)
