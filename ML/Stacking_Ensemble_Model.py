import numpy as np
from sklearn.model_selection import StratifiedKFold

def stacking_model(model , trainX , trainY , test , n_folds = 5):
    kfold = StratifiedKFold(n_splits = n_folds , random_state = 42)
    
    train_predict = np.zeros((trainX.shape[0] , 1))
    test_predict = np.zeros((test.shape[0] , n_folds))
    print("Model : " , model.__class__.__name__)
    
    for i , (train_index , valid_index) in enumerate(kfold.split(trainX)):
        X = trainX[train_index]
        y = trainY[train_index]
        X_validation = trainX[valid_index]
        
        model.fit(X , y)
        train_predict[valid_index , :] = model.predict(X_validation).reshape(-1 , 1)
        test_predict[: , i] = model.predict(test)
        
    test_predict_mean = np.mean(test_predict , axis = 1).reshape(-1 , 1)
    
    return train_predict , test_predict_mean