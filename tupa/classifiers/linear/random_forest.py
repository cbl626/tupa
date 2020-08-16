import time
from collections import defaultdict, Counter
from itertools import repeat
import numpy as np
#from tupa.model_util import KeyBasedDefaultDict, save_dict, load_dict
#from ..classifier import Classifier
from classifiers.classifier import Classifier
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import re
import bz2
import pickle
import _pickle as cPickle
from scipy.sparse import coo_matrix


# =========== FUNCTIONS FOR DATA PREPROCESSING ============ #

# Saves and loads files as pickle in compressed format:
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


# Carries out one-hot encoding of the dataset and saves the result in a coordinate matrix.
def build_dataset(list_of_vectors, feature_vector):
    idx_list = []

    for vector in list_of_vectors:
        unique_vals = set(vector)
        new_row = [val in unique_vals for val in feature_vector]
        new_idx = np.where(new_row)[0]
        idx_list.append(new_idx)

    all_rows = []
    all_cols = []
    all_data = []

    for i in range(len(idx_list)):
        cols = idx_list[i]
        rows = [i]*cols.size
        data = [True]*cols.size

        all_rows.extend(rows)
        all_cols.extend(cols.tolist())
        all_data.extend(data)

    matrix = coo_matrix((all_data, (all_rows, all_cols)), shape=(len(list_of_vectors), len(feature_vector)), dtype='uint8')
    return matrix


# Transforms the list of labels from objects into strings, so that 
# they can be used as label data alongside the coo_matrix training data.
def label_objects_to_string(input_labels):
    output_labels = []
    for number in input_labels:
        output_labels.append(str(number))
    return output_labels


# Creates a feature vector consisting of all unique labels in the training dataset.
# Can also optionally remove x number of the least frequent features from the feature vector,
# which can drastically reduce computation time, and possibly with minimal loss
# of prediction accuracy.
def create_feature_vector(input_vectors, keep_x_most_frequent):
    features_sorted_by_count = Counter(x for xs in input_vectors for x in set(xs)).most_common()
    reduced_feature_vector = features_sorted_by_count[:keep_x_most_frequent]
    feature_vector = []
    for (i,_) in reduced_feature_vector:
        feature_vector.append(i)
    return feature_vector


# Builds a vector consisting of only one of each unique transition label (in order of appearance).
def create_unique_labels_vector(train_data_labels):
    train_data_labels = pd.Series(train_data_labels)
    train_data_labels = train_data_labels.astype(str)
    unique_labels = train_data_labels.unique()
    tmp_list = []
    for val in unique_labels:
        for number in re.findall(r'\d+', val):
            tmp_list.append(number)
    unique_labels = pd.unique(tmp_list)
    return unique_labels


# Transforms prediction label to scores format.
def pred_to_scores(y_pred, unique_labels):
    scores = np.zeros([len(unique_labels)])
    y_pred = str(y_pred)
    y_pred = y_pred.strip("[']")
    if (len(y_pred) > 2):
        y_pred = y_pred.split(',', 1)[0]
    for idx, val in enumerate(unique_labels):
        if (val == y_pred):
            scores[idx] = 1
    scores = scores.astype(float)
    return scores


class RandomForest(Classifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = RandomForestClassifier(bootstrap=True,
                                            class_weight=None,
                                            criterion='gini',
                                            max_depth=10,
                                            max_features='auto',
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_samples_leaf=1,
                                            min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            n_estimators=100,
                                            n_jobs=-1,
                                            oob_score=False,
                                            random_state=None,
                                            verbose=0,
                                            warm_start=False)
        self.training = True
        self.list_of_TRAIN_vectors = []
        self.list_of_TRAIN_labels = []


    def update(self, features, axis, pred, true, importance=None):
        self.list_of_TRAIN_vectors.append([*features])
        self.list_of_TRAIN_labels.append(true)


    def score(self, features, axis):
        if (self.training == True):
            super().score(features, axis)
            return np.zeros(self.num_labels[axis])
        else:
            super().score(features, axis)
            x_test = coo_matrix([val in set([*features]) for val in self.feature_vector])
            y_pred = self.model.predict(x_test)
            scores = pred_to_scores(y_pred, self.unique_labels)
            return scores


    def resize(self, *args, **kwargs):
        pass


    def finalize(self, finished_epoch=False, **kwargs):
        super().finalize(finished_epoch=finished_epoch, **kwargs)
        if finished_epoch:
            self.training = False

            print("Building the feature vector…")
            keep_x_most_frequent = 10000 # <-- Adjustable. If this number is higher than the number of unique features, it will include all features.
            self.feature_vector = create_feature_vector(self.list_of_TRAIN_vectors, keep_x_most_frequent)

            print("Transforming the label data from objects to strings...")
            self.list_of_TRAIN_labels = label_objects_to_string(self.list_of_TRAIN_labels)

            print("Building the unique labels vector…")
            self.unique_labels = create_unique_labels_vector(self.list_of_TRAIN_labels)

            print("Building the dataset…")
            self.train_data = build_dataset(self.list_of_TRAIN_vectors, self.feature_vector)
            print("The shape of the dataset:", self.train_data.shape)

            print("The Random Forest Classifier is fitting the data…")
            y_train = self.list_of_TRAIN_labels
            x_train = self.train_data
            self.model.fit(x_train,y_train)
            print("The Random Forest Classifier has finished fitting the data!")

            return self