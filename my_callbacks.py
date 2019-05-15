from keras.callbacks import Callback
from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
import os
import matplotlib.pyplot as plt
import shutil
import numpy as np

class PerSubjAucMetricHistory(Callback):
    """
    This callback for testing model on each subject separately during training. It writes auc for every subject to the
    history object
    """
    def __init__(self,subjects):
        self.subjects = subjects
        super(PerSubjAucMetricHistory,self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        for subj in self.subjects.keys():
            x,y = self.subjects[subj]

            y_pred = self.model.predict(x, verbose=0)
            if isinstance(y_pred,list):
                y_pred = y_pred[0]
            if len(y_pred.shape) == 1:
                y_pred = to_categorical(y_pred,2)
            if len(y.shape) == 1:
                y = to_categorical(y,2)
            y_pred = to_categorical(y_pred) if (len(y_pred.shape)==1) else y_pred
            logs['val_auc_%s' %(subj)] = roc_auc_score(y[:,1], y_pred[:,1])
            if type(self.model.output) == list:
                fake_subj_labels = np.zeros((len(y),self.model.output[1].shape._dims[1]._value))
                logs['val_loss_%s' % (subj)] = self.model.evaluate(x,[y,fake_subj_labels], verbose=0)[0]
            else:
                logs['val_loss_%s' % (subj)] = self.model.evaluate(x,y, verbose=0)[0]

class AucMetricHistory(Callback):
    def __init__(self,save_best_by_auc=False,path_to_save=None):
        super(AucMetricHistory, self).__init__()
        self.save_best_by_auc=save_best_by_auc
        self.path_to_save = path_to_save
        self.best_auc = 0
        self.best_epoch = 1
        if save_best_by_auc and (path_to_save is None):
            raise ValueError('Specify path to save the model')

    def on_epoch_end(self, epoch, logs={}):
        x_val,y_val = self.validation_data[0],self.validation_data[1]
        y_pred = self.model.predict(x_val,batch_size=len(y_val), verbose=0)
        if isinstance(y_pred,list):
            y_pred = y_pred[0]
        current_auc = roc_auc_score(y_val, y_pred)
        logs['val_auc'] = current_auc

        if current_auc > self.best_auc:
            if self.save_best_by_auc:
                prev_model_path = os.path.join(self.path_to_save,'best_on_auc_%d_%.2f.hdf5' %(self.best_epoch,self.best_auc))
                if os.path.isfile(prev_model_path):
                    os.remove(prev_model_path)

                path_to_file = os.path.join(self.path_to_save, 'best_on_auc_%d_%.2f.hdf5' % (epoch,current_auc))
                self.model.save(path_to_file)

            self.best_auc = current_auc
            self.best_epoch = epoch

class DomainActivations(Callback):
    def __init__(self, x_train,y_train, subj_label_train,path_to_save):
        super(DomainActivations, self).__init__()
        self.path_to_save = '%s/domain_activations_grl/' % path_to_save
        self.x_train = x_train
        self.y_train = y_train
        self.subj_label_train = subj_label_train
        plt.plot(subj_label_train.argmax(axis=1))
        if os.path.isdir(self.path_to_save):
            shutil.rmtree(self.path_to_save)
        os.makedirs(self.path_to_save)
        plt.savefig(os.path.join('%s/class_distr' % self.path_to_save))
        plt.close()
    def _log_domain_activations(self, domain_label_pred, domain_label,pic_name):
        activations = (domain_label_pred * domain_label).sum(axis=1)
        plt.plot(activations)
        # plt.plot(activations[self.y_train[:,1] == 1])
        plt.savefig(os.path.join('%s/%s' % (self.path_to_save, pic_name)))
        plt.close()

    def on_epoch_end(self, epoch, logs=None):
        if epoch %10 ==0:
            self._log_domain_activations(self.model.predict(self.x_train, verbose=0)[1],self.subj_label_train,'%d_train' % epoch)



