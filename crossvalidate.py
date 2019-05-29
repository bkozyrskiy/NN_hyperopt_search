import os
from sklearn.model_selection import StratifiedKFold
from my_callbacks import AucMetricHistory
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model
from utils import single_auc_loging


def crossvalidate(x_train_val, y_train_val, model, model_path,epochs):
    path_to_clean_weights = os.path.join(model_path,'tmp.h5')
    os.makedirs(model_path)
    model.save_weights(path_to_clean_weights)  # Nasty hack. This weights will be used to reset model
    folds = 4  # To preserve split as 0.6 0.2 0.2
    cv = StratifiedKFold(n_splits=folds, shuffle=True)
    best_val_epochs = []
    best_val_aucs = []
    val_histories=[]
    for fold, (train_idx, val_idx) in enumerate(cv.split(x_train_val, y_train_val[:, 1])):
        # for fold, (train_idx, val_idx) in enumerate(cv.split(x_tr, y_tr)):
        fold_model_path = os.path.join(model_path, '%d' % fold)
        os.makedirs(fold_model_path)
        same_subj_auc = AucMetricHistory(save_best_by_auc=True, path_to_save=fold_model_path)
        # make_checkpoint = ModelCheckpoint(os.path.join(fold_model_path, '{epoch:02d}.hdf5'),
        #                                   monitor='val_loss', verbose=0, save_best_only=False, mode='auto')

        x_tr_fold, y_tr_fold = x_train_val[train_idx], y_train_val[train_idx]
        x_val_fold, y_val_fold = x_train_val[val_idx], y_train_val[val_idx]
        val_history = model.fit(x_tr_fold, y_tr_fold, epochs=epochs, validation_data=(x_val_fold, y_val_fold),
                                callbacks=[same_subj_auc], batch_size=64, shuffle=True)
        best_val_epochs.append(np.argmax(val_history.history['val_auc']) + 1)  # epochs count from 1 (not from 0)
        best_val_aucs.append(np.max(val_history.history['val_auc']))
        model.load_weights(path_to_clean_weights)  # Rest model after each fold
        single_auc_loging(val_history.history,'fold_history',fold_model_path)
    os.remove(path_to_clean_weights)

    return best_val_aucs, best_val_epochs,val_history

def test_ensamble(x_tst,y_tst,model_path):
    predictions = np.zeros_like(y_tst)
    folds = 0
    for fold_folder in os.listdir(model_path):
        fold_model_path = os.path.join(model_path, fold_folder)
        if (os.path.isdir(fold_model_path)) and (fold_folder.isdigit()):
            folds += 1
            model_checkpoint = next(filter(lambda x:  os.path.splitext(x)[1]=='.hdf5',os.listdir(fold_model_path)))
            fold_model_path = os.path.join(fold_model_path, model_checkpoint)
            predictions += np.squeeze(load_model(fold_model_path).predict(x_tst))

    predictions /= (folds)
    test_auc_ensemble = roc_auc_score(y_tst, predictions)
    return test_auc_ensemble

def test_naive(x_train_val, y_train_val,x_tst,y_tst,model,num_epochs,model_path):
    '''

    :param x_train_val:
    :param y_train_val:
    :param x_tst:
    :param y_tst:
    :param model: Be ESTREAMLY careful with it. Check, that it's CLEAN!
    :param num_epochs:
    :return:
    '''
     # Rest model before traning on train+val
    same_subj_auc = AucMetricHistory(save_best_by_auc=False)
    test_history = model.fit(x_train_val, y_train_val, epochs=int(num_epochs),
                        validation_data=(x_tst, y_tst),callbacks=[same_subj_auc],batch_size=64, shuffle=True)
    model.save(os.path.join(model_path,'final_%d.hdf5' %int(num_epochs)))
    single_auc_loging(test_history.history, 'naive_test_history', model_path)
    return test_history.history






