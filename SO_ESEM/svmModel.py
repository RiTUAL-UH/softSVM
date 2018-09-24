
from sklearn import svm
import random
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import pandas as pd



def run_SVM():
    """
    The main function for running the model; using cosine and soft-cosine similarity features
    :return:
    """
    feature_number = [0,1,2,3,4,5,6, 7, 8, 9]
    dir = "features/hand-feats/"

    train_feats_body = np.load(open(dir + "train_all_feats_noClue_medium.npy", "rb"))
    test_feats_body =np.load(open(dir + "test_all_feats_noClue_medium.npy", "rb"))

    print("shape body train: ", train_feats_body.shape)

    print("shape body test: ", test_feats_body.shape)

    print("train:")
    print(train_feats_body[:,feature_number])

    print("train + dev: ")
    body = np.column_stack((
                                train_feats_body[:, feature_number[6]],
                           train_feats_body[:, feature_number[7]],
                             train_feats_body[:, feature_number[8]],
                             train_feats_body[:, feature_number[9]]))

    print("body shape: ")
    print(body.shape)
    print()

    train_concat = body
    print()
    test_body = np.column_stack((
                                            test_feats_body[:, feature_number[6]]
                                           ,test_feats_body[:, feature_number[7]]
                                           ,test_feats_body[:, feature_number[8]],
                                           test_feats_body[:, feature_number[9]]))

    test_concat = test_body

    train_labels,test_labels = label_fixer()
    print("feature number: ", feature_number)

    print("train lable size: ", len(train_labels))
    print("train shape ", train_concat.shape)
    print("train labels ", train_labels[:5])
    print("true labels: ", test_labels[:5])

    clf = svm.SVC()
    clf.fit(train_concat, train_labels)
    predicted = clf.predict(test_concat)
    #
    print("train labels ", train_labels[:5])
    print("predicted: ", predicted[:5])
    print(type(predicted))
    print("true labels: ", test_labels[:5])
    print(type(test_labels))
    #

    print("features 6 7 8 9 ")
    print("f1_score: {}".format(f1_score(test_labels, predicted, average="micro")))
    print(classification_report(test_labels, predicted, target_names=["1", "2", "3", "4"]))







def label_fixer():
    """
    This function fix the formatting of the labels
    :return: fixed labels for test and train
    """
    dir = "features/hand-feats/"
    train_labels = np.load(open(dir+"train_labels_FILE.npy", "rb"))

    test_labels = np.load(open(dir+"test_labels_FILE.npy", "rb"))
    fixed_train_labels=[]
    fixed_test_labels = []
    print("train label size: ", train_labels.shape)

    #train:

    print("@@@@@train labels: ", train_labels[:10])


    for i in train_labels:
        if i[0] == 1:
            fixed_train_labels.append(1)
        elif i[1] == 1:
            fixed_train_labels.append(2)
        elif i[2] == 1:
            fixed_train_labels.append(3)
        elif i[3] == 1:
            fixed_train_labels.append(4)

    print("@@@@@@@@@@")
    print("train_lable: ", len(fixed_train_labels))
    # test:


    for i in test_labels:
        if i[0] == 1:
            fixed_test_labels.append(1)
        elif i[1] == 1:
            fixed_test_labels.append(2)
        elif i[2] == 1:
            fixed_test_labels.append(3)
        elif i[3] == 1:
            fixed_test_labels.append(4)

    print("shape of fixed: , ", fixed_test_labels[:10])
    return fixed_train_labels, fixed_test_labels



if __name__ == '__main__':
    run_SVM()

