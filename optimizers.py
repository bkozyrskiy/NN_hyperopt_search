import GPy
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import pickle
from hyperopt import hp, tpe, fmin, Trials
from functools import partial
import os

def run_a_trial_hp(subjects, subj_tr_val_ind, subj_tst_ind, results_dir, f_to_optimize, space):
    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 1
    trials_path = os.path.join(results_dir,"trials.pkl")
    print("Attempt to resume a past training if it exists:")
    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open(trials_path, "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    function_to_opt = partial(
        f_to_optimize,
        subjects=subjects,
        subj_tr_val_ind = subj_tr_val_ind,
        subj_tst_ind = subj_tst_ind)
    best = fmin(
        function_to_opt,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open(trials_path, "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")
    return best

def run_gp(subjects, subj_tr_val_ind, subj_tst_ind, f_to_optimize, space, noise_var,max_iter):
    def f2opt_wrapper(params):
        '''Used to transform nparray to dict'''
        params_dict = {}
        for elem_idx, elem in enumerate(space):
            params_dict[elem['name']] = params[0,elem_idx]
        return f_to_optimize(params_dict,subjects=subjects, subj_tr_val_ind=subj_tr_val_ind, subj_tst_ind=subj_tst_ind)

    kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
    optimizer = BayesianOptimization(f=f2opt_wrapper,
                                     domain=space,
                                     model_type='GP',
                                     kernel=kernel,
                                     acquisition_type='EI',
                                     acquisition_jitter=0.05,
                                     noise_var = noise_var,
                                     exact_feval=False,
                                     maximize=True,
                                     verbose=True)
    optimizer.run_optimization(max_iter=max_iter,verbosity=True)
    return optimizer.x_opt