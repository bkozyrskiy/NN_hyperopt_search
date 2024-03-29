from hyperopt import hp, STATUS_OK


import numpy as np
from mne.filter import resample
from crossvalidate import crossvalidate,test_ensamble,test_naive, run_a_trial
from keras.utils import to_categorical
import keras.backend as K
import uuid
from utils import save_results,get_subj_split
from my_models import ShallowConvNet
import os
import sys
sys.path.append(os.path.join(os.path.split(os.getcwd())[0],'data_loader'))

from data import DataBuildClassifier,EEG_SAMPLE_RATE

RESULTS_DIR = "results_shallow/"
WEIGHTS_DIR = "weights_shallow/"

space = {'resample_to': hp.choice('resample_to', range(128, 501)),
             'dropoutRate': hp.uniform('dropoutRate', 0, 1),
             'lr': hp.loguniform('lr', -5 * np.log(10), -3 * np.log(10))
             }

def build_and_train_all_subjects(params,subjects,subj_tr_val_ind,subj_tst_ind):
    params_uuid = str(uuid.uuid4())[:5]
    subj_val_aucs,subj_tst_aucs_ens,subj_tst_aucs_naive = {},{},{}
    tmp_weights_res_path = os.path.join(WEIGHTS_DIR,params_uuid)
    # for subj in subjects.keys():
    for subj in [25,26]:
        K.clear_session()
        tr_val_ind = subj_tr_val_ind[subj]
        tst_ind = subj_tst_ind[subj]
        x_tr_val,y_tr_val = subjects[subj][0][tr_val_ind], to_categorical(subjects[subj][1][tr_val_ind],2)
        x_tst, y_tst = subjects[subj][0][tst_ind], to_categorical(subjects[subj][1][tst_ind],2)

        x_tr_val = resample(x_tr_val, up=1., down=EEG_SAMPLE_RATE/params['resample_to'], npad='auto', axis=1)
        x_tst = resample(x_tst, up=1., down=EEG_SAMPLE_RATE / params['resample_to'], npad='auto', axis=1)

        model_path = os.path.join(tmp_weights_res_path,str(subj))

        model = ShallowConvNet(params,Chans=x_tr_val.shape[2], Samples=x_tr_val.shape[1])
        x_tr_val = x_tr_val.transpose(0, 2, 1)[:,np.newaxis,:,:]
        x_tst = x_tst.transpose(0, 2, 1)[:, np.newaxis, :, :]
        val_aucs, val_aucs_epochs,_  = crossvalidate(x_tr_val, y_tr_val, model, model_path)

        test_auc_ensemble = test_ensamble(x_tst,y_tst,model_path)
        test_naive_history = test_naive(x_tr_val, y_tr_val, x_tst, y_tst, model, int(np.mean(val_aucs_epochs)), model_path)
        test_auc_naive = test_naive_history['val_auc'][-1]

        subj_val_aucs[subj] = np.mean(val_aucs)
        subj_tst_aucs_ens[subj] = test_auc_ensemble
        subj_tst_aucs_naive[subj] = test_auc_naive

    median_val_aucs = np.median(list(subj_val_aucs.values()))
    weights_res_path = os.path.join(WEIGHTS_DIR, '%.2f_%s' % (median_val_aucs,params_uuid))
    os.rename(tmp_weights_res_path,weights_res_path)

    params_res_path = os.path.join(RESULTS_DIR, '%.2f_%s' % (median_val_aucs,params_uuid))
    save_results(params_res_path, subj_val_aucs,subj_tst_aucs_naive, subj_tst_aucs_ens, params)
    result= {
        'loss': -median_val_aucs,
        'real_loss': np.mean(list(subj_tst_aucs_naive.values())),
        'subj_tst_aucs_naive':subj_tst_aucs_naive,
        'subj_tst_aucs_ens':subj_tst_aucs_ens,
        'subj_val_aucs':subj_val_aucs,
        'status': STATUS_OK
    }

    return result




if __name__ == '__main__':
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
    data = DataBuildClassifier('/home/likan_blk/BCI/NewData')
    subjects, subj_tr_val_ind, subj_tst_ind = get_subj_split(data)
    # split_subj = lambda x, ind: {key: (x[key][0][ind[key]], x[key][1][ind[key]]) for key in x}
    # subj_train_val = split_subj(subjects,subj_tr_val_ind)
    # subj_test = split_subj(subjects, subj_tst_ind)
    for t in range(3):
        run_a_trial(subjects, subj_tr_val_ind, subj_tst_ind,RESULTS_DIR,build_and_train_all_subjects,space)