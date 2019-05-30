import codecs
from bson import json_util
import numpy as np
import json
import os
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import operator
import shutil

def read_conifig():
    with open('config.json') as f:
        config = json.load(f)
    return config

def save_json_result(res_dir,params):
    """Save json to a directory and a filename."""
    result_name = '{}.txt.json'.format('hyperparams')
    with open(os.path.join(res_dir, result_name), 'w') as f:
        json.dump(
            params, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )

def write_results_table(params_res_path,subj_val_aucs, subj_tst_aucs_naive, subj_tst_aucs_ens):
    with codecs.open('%s/res.txt' %params_res_path,'w', encoding='utf8') as f:
        f.write('subj,mean_val_aucs, test_aucs_naive, test_aucs_ensemble\n')
        for tr_subj_idx, tr_subj in enumerate(subj_val_aucs.keys()):
            f.write(u'%d, %.04f, %.04f, %.04f\n' \
                    % (tr_subj,subj_val_aucs[tr_subj], subj_tst_aucs_naive[tr_subj],subj_tst_aucs_ens[tr_subj]))

        mean_val_auc = np.mean(list(subj_val_aucs.values()))
        std_val_auc = np.std(list(subj_val_aucs.values()))
        mean_tst_auc_ens = np.mean(list(subj_tst_aucs_ens.values()))
        std_tst_auc_ens = np.std(list(subj_tst_aucs_ens.values()))
        mean_tst_auc_naive = np.mean(list(subj_tst_aucs_naive.values()))
        std_tst_auc_naive = np.std(list(subj_tst_aucs_naive.values()))

        f.write(u'MEAN, %.04f±%.04f, %.04f±%.04f, %.04f±%.04f\n' \
                % (mean_val_auc, std_val_auc, mean_tst_auc_naive, std_tst_auc_naive, mean_tst_auc_ens, std_tst_auc_ens))

        median_val_auc = np.median(list(subj_val_aucs.values()))
        median_tst_auc_ens = np.median(list(subj_tst_aucs_ens.values()))
        median_tst_auc_naive = np.median(list(subj_tst_aucs_ens.values()))
        f.write(u'MEDIAN, %.04f, %.04f, %.04f\n' \
                % (median_val_auc, median_tst_auc_naive, median_tst_auc_ens))


def save_results(res_dir,subj_val_aucs,subj_tst_aucs_ens,subj_tst_aucs_naive,params):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    write_results_table(res_dir,subj_val_aucs, subj_tst_aucs_naive, subj_tst_aucs_ens)
    save_json_result(res_dir,params)


def get_subj_split(data,subj_numbers):
    subj_tr_val_ind = {}
    subj_tst_ind = {}
    if os.path.exists('subj_tr_val_ind.pkl') and os.path.exists('subj_tst_ind.pkl'):
        with open('subj_tr_val_ind.pkl', 'rb') as f:
            subj_tr_val_ind = pickle.load(f)
        with open('subj_tst_ind.pkl', 'rb') as f:
            subj_tst_ind = pickle.load(f)

    subjects = data.get_data(subj_numbers, shuffle=False, windows=[(0.2, 0.5)], baseline_window=(0.2, 0.3))
    if (len(subj_tr_val_ind) == 0) and (len(subj_tst_ind) == 0):
        for subj in subjects.keys():
            x = subjects[subj][0]
            y = to_categorical(subjects[subj][1], 2)
            x_tr_val_ind, x_tst_ind, y_tr_val, y_tst = train_test_split(range(x.shape[0]), y, test_size=0.2, stratify=y)
            subj_tr_val_ind[subj] = (x_tr_val_ind)
            subj_tst_ind[subj] = (x_tst_ind)

        with open('subj_tst_ind.pkl', 'wb') as f:
            pickle.dump(subj_tst_ind, f, pickle.HIGHEST_PROTOCOL)

        with open('subj_tr_val_ind.pkl', 'wb') as f:
            pickle.dump(subj_tr_val_ind, f, pickle.HIGHEST_PROTOCOL)


    return subjects, subj_tr_val_ind, subj_tst_ind

def single_auc_loging(history,title,path_to_save):
    """
    Function for ploting nn-classifier performance. It makes two subplots.
    First subplot with train and val losses
    Second with val auc
    Function saves plot as a picture and as a pkl file

    :param history: history field of history object, witch returned by model.fit()
    :param title: Title for picture (also used as filename)
    :param path_to_save: Path to save file
    :return:
    """
    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,12))

    if 'loss' in history.keys():
        loss_key = 'loss'  # for simple NN
    elif 'class_out_loss' in history.keys():
        loss_key = 'class_out_loss'  # for DAL NN
    else:
        raise ValueError('Not found correct key for loss information in history')

    ax1.plot(history[loss_key],label='cl train loss')
    ax1.plot(history['val_%s' %loss_key],label='cl val loss')
    ax1.legend()
    min_loss_index,max_loss_value = min(enumerate(history['val_loss']), key=operator.itemgetter(1))
    ax1.set_title('min_loss_%.3f_epoch%d' % (max_loss_value, min_loss_index))
    ax2.plot(history['val_auc'])
    max_auc_index, max_auc_value = max(enumerate(history['val_auc']), key=operator.itemgetter(1))
    ax2.set_title('max_auc_%.3f_epoch%d' % (max_auc_value, max_auc_index))
    f.suptitle('%s' % (title))
    plt.savefig('%s/%s.png' % (path_to_save,title), figure=f)
    plt.close()
    with open('%s/%s.pkl' % (path_to_save,title), 'wb') as output:
        pickle.dump(history,output,pickle.HIGHEST_PROTOCOL)


def clear_res_weight_dir(res_dir,weights_dir):
    """Function remove all dirs except the best by first value in dir name (VAL AUC value)"""
    res_run_dirs = os.listdir(res_dir)
    max_val_auc = 0
    best_dir = None
    for dir_name in res_run_dirs:
        val_auc = float(dir_name.split('_')[0])
        if val_auc > max_val_auc:
            best_dir = dir_name
    for dir in res_run_dirs:
        if dir != best_dir:
            shutil.rmtree(os.path.join(res_dir,dir))
            shutil.rmtree(os.path.join(weights_dir, dir))