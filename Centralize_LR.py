from sklearn.linear_model import LogisticRegressionCV
import utils

#Centralize Logistic Regression: RBC, BMO, TD, CICB, Soctia


#0.Load Data
#load train data
(train_x, train_y, train_all) = utils.load_credit_train_Default()
#load test data
(test_x, test_y,y_test_p, test_all) = utils.load_credit_test_Default()
#random state for spiling
random_state = 20211217
#split train/test data into 4 partition
(train_sp1,train_sp2,train_sp3,train_sp4) = utils.split_data_4(train_all,"train",random_state)
#split test data into 4 partition
(train_sp1,train_sp2,train_sp3,train_sp4) = utils.split_data_4(test_all,"test",random_state)


#1.Centrealize Logistic Regression for RBC
# model
clf_all = LogisticRegressionCV(cv=5, random_state=0,
                             penalty="l2",
                             max_iter=utils.load_parameter_C()[0],  # local epoch
                             Cs=utils.load_parameter_C()[1],
                             ).fit(train_x, train_y)
#predict
predict_all = clf_all.predict(test_x)
#accuracy
accuracy = utils.accuray_evaluation(predict_all,test_y)
print("RBC:")
print(accuracy)


#2.Centrealize Logistic Regression for BMO, TD, CICB, Soctia:
#Loop
for i in range(1,5):
    #load train and test data
    (train_x_1, train_y_1) = utils.load_credit_train_i_Default(i)
    (x_test_1, y_test_1, y_test_p_1) = utils.load_credit_test_i_Default(i)
    #model
    clf_all_1 = LogisticRegressionCV(cv=5, random_state=0,
                                   penalty="l2",
                                   max_iter=utils.load_parameter_C()[0],  # local epoch
                                   # warm_start=True,  # prevent refreshing weights when fitting
                                   Cs=utils.load_parameter_C()[1]
                                   ).fit(train_x, train_y)
    #predict
    predict_all_1 = clf_all_1.predict(x_test_1)
    accuracy_1 = utils.accuray_evaluation(predict_all_1, y_test_1)
    # #Profit
    j = str(i)
    print("Client_" + j + ":")
    print("Accuracy_own_data_own_model:")
    print(accuracy_1)
