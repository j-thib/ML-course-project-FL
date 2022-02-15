from typing import Tuple, Union, List
import pandas as pd
from numpy.core import ravel
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import numpy as np

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]
#Load Data
train = pd.read_csv("credit/CreditGame_TRAIN_all.csv")
test = pd.read_csv("credit/CreditGame_TEST_all.csv")
train_1 = pd.read_csv("credit/CreditGame_train1.csv")
train_2 = pd.read_csv("credit/CreditGame_train2.csv")
train_3 = pd.read_csv("credit/CreditGame_train3.csv")
train_4 = pd.read_csv("credit/CreditGame_train4.csv")
test_1 = pd.read_csv("credit/CreditGame_test1.csv")
test_2 = pd.read_csv("credit/CreditGame_test2.csv")
test_3 = pd.read_csv("credit/CreditGame_test3.csv")
test_4 = pd.read_csv("credit/CreditGame_test4.csv")

def get_model_parameters(model):
    """Returns the paramters of a sklearn LogisticRegression model"""
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_,)
    return params

def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model: LogisticRegression):
    """
    Sets initial parameters as zeros
    """
    n_classes = 2 # MNIST has 10 classes
    n_features = 27 # Number of features in dataset
    model.classes_ = np.array([i for i in range(2)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

def load_credit_test_Default() -> Dataset:
    """
    Loads the Credit dataset
    """
    col_n_y = ['DEFAULT']
    y_test = pd.DataFrame(test, columns=col_n_y)

    col_n_x = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL', 'MNT_EPAR',
               'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M', 'NB_INTR_12M', 'PIR_DEL',
               'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT', 'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI',
               'MNT_DEMANDE']

    x_test = pd.DataFrame(test, columns=col_n_x)

    col_n_y_k = ['PROFIT_LOSS']
    y_test_p = pd.DataFrame(test, columns=col_n_y_k)

    test_all = pd.DataFrame(test)

    return (x_test, y_test,y_test_p, test_all)

def load_credit_test_i_Default(i) -> Dataset:
    """
    Loads the Credit dataset
    """
    i = str(i)

    test_i = pd.read_csv("credit/CreditGame_test"+i+".csv")

    col_n_y = ['DEFAULT']
    y_test_1 = pd.DataFrame(test_i, columns=col_n_y)

    col_n_x = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL', 'MNT_EPAR',
               'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M', 'NB_INTR_12M', 'PIR_DEL',
               'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT', 'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI',
               'MNT_DEMANDE']
    x_test_1 = pd.DataFrame(test_i, columns=col_n_x)

    col_n_y_k = ['PROFIT_LOSS']
    y_test_p_1 = pd.DataFrame(test_i, columns=col_n_y_k)


    return (x_test_1, y_test_1,y_test_p_1)


def load_credit_test_Profit() -> Dataset:
    """
    Loads the Credit dataset
    """
    col_n_y_k = ['PROFIT_LOSS']
    y_test_1 = pd.DataFrame(test, columns=col_n_y_k)

    col_n_x_k = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL', 'MNT_EPAR',
               'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M', 'NB_INTR_12M', 'PIR_DEL',
               'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT', 'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI',
               'MNT_DEMANDE']
    x_test_1 = pd.DataFrame(test, columns=col_n_x_k)

    return (x_test_1, y_test_1)

def load_credit_train_Default() -> Dataset:
    """
    Loads the Credit dataset
    """
    col_n_y_1 = ['DEFAULT']
    y_train = pd.DataFrame(train, columns=col_n_y_1)

    col_n_x_1 = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL', 'MNT_EPAR',
               'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M', 'NB_INTR_12M', 'PIR_DEL',
               'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT', 'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI',
               'MNT_DEMANDE']


    x_train = pd.DataFrame(train, columns=col_n_x_1)

    col_n_z_1 = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL', 'MNT_EPAR',
               'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M', 'NB_INTR_12M', 'PIR_DEL',
               'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT', 'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI',
               'MNT_DEMANDE','DEFAULT','PROFIT_LOSS']

    train_all = pd.DataFrame(train,columns=col_n_z_1)

    return (x_train, y_train, train_all)


def load_credit_train_i_Default(i) -> Dataset:
    """
    Loads the Credit dataset
    """

    i = str(i)

    train_i = pd.read_csv("credit/CreditGame_train"+i+".csv")

    col_n_y_1 = ['DEFAULT']
    y_train_1 = pd.DataFrame(train_i, columns=col_n_y_1)

    col_n_x_1 = ['NB_EMPT', 'R_ATD', 'DUREE', 'PRT_VAL', 'AGE_D', 'REV_BT', 'REV_NET', 'TYP_RES', 'ST_EMPL', 'MNT_EPAR',
               'NB_ER_6MS', 'NB_ER_12MS', 'NB_DEC_12MS', 'NB_OPER', 'NB_COUR', 'NB_INTR_1M', 'NB_INTR_12M', 'PIR_DEL',
               'NB_DEL_30', 'NB_DEL_60', 'NB_DEL_90', 'MNT_PASS', 'MNT_ACT', 'MNT_AUT_REN', 'MNT_UTIL_REN', 'NB_SATI',
               'MNT_DEMANDE']
    x_train_1 = pd.DataFrame(train_i, columns=col_n_x_1)

    return (x_train_1, y_train_1)

def load_parameter_C():
    """
    Loads the parameter
    """
    #client iteration
    max_iter = 10
    #server round
    round = 20
    Cs = [0.01, 0.1, 1, 10]

    return (max_iter,round,Cs)

def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    print(num_partitions)
    return list(
        zip(np.array_split(X, num_partitions),
        np.array_split(y, num_partitions))

    )

#split data into 4 partition
def split_data_4(data, type,random_state):
    np.random.seed(random_state)
    shuffled_indice = np.random.permutation(len(data))
    data_size_1 = int(len(data) * 0.25)
    data_size_2 = int(len(data) * 0.5)
    data_size_3 = int(len(data) * 0.75)

    train_inices_1 = shuffled_indice[:data_size_1]
    train_inices_2 = shuffled_indice[data_size_1:data_size_2]
    train_inices_3 = shuffled_indice[data_size_2:data_size_3]
    train_inices_4 = shuffled_indice[data_size_3:]

    #save
    data_1 = pd.DataFrame(data.iloc[train_inices_1])
    data_2 = pd.DataFrame(data.iloc[train_inices_2])
    data_3 = pd.DataFrame(data.iloc[train_inices_3])
    data_4 = pd.DataFrame(data.iloc[train_inices_4])

    data_1.to_csv("credit/CreditGame_" + type + "1.csv",header=True)
    data_2.to_csv("credit/CreditGame_" + type + "2.csv", header=True)
    data_3.to_csv("credit/CreditGame_" + type + "3.csv", header=True)
    data_4.to_csv("credit/CreditGame_" + type + "4.csv", header=True)

    return data_1,data_2,data_3,data_4

def profit_evaluation(predict_all,y_test_p):
    for i in range(0, len(predict_all)):
        if predict_all[i - 1] == 1:
            y_test_p[i:i + 1] = 0
    profit = sum(y_test_p['PROFIT_LOSS'])

    return profit

def kmeans_split_data_4(data, type,random_state):
    kmeans = KMeans(n_clusters=4, random_state=random_state,max_iter=10).fit(data)

    #save
    data_1 = pd.DataFrame(data[kmeans.labels_==1])
    data_2 = pd.DataFrame(data[kmeans.labels_==2])
    data_3 = pd.DataFrame(data[kmeans.labels_==3])
    data_4 = pd.DataFrame(data[kmeans.labels_==4])

    data_1.to_csv("credit/CreditGame_" + type + "1.csv",header=True)
    data_2.to_csv("credit/CreditGame_" + type + "2.csv", header=True)
    data_3.to_csv("credit/CreditGame_" + type + "3.csv", header=True)
    data_4.to_csv("credit/CreditGame_" + type + "4.csv", header=True)

    return data_1,data_2,data_3,data_4

def accuray_evaluation(predict_all,test_y):
    accuracy = np.mean(predict_all == ravel(test_y))

    return accuracy

def load_credit_train_test_Default(random_state):
    data = pd.DataFrame(train_all_all)
    np.random.seed(random_state)
    shuffled_indice = np.random.permutation(round(len(data)*0.4))
    data_size_1 = int(len(data) * 0.8)

    train_inices_1 = shuffled_indice[:data_size_1]
    train_inices_2 = shuffled_indice[data_size_1:]

    #save
    data_1 = pd.DataFrame(data.iloc[train_inices_1])
    data_2 = pd.DataFrame(data.iloc[train_inices_2])
    data_1.to_csv("credit/CreditGame_TRAIN_all.csv",header=True)
    data_2.to_csv("credit/CreditGame_TEST_all.csv", header=True)

    return data_1, data_2