from train_gl_jsd_without_mixup import main
from utils.hyper_param_util import setup_args
from utils.misc import unset_random_seed
import optuna
import time
import sys
from datetime import datetime


def objective(trial):
    print('\nRunning Trial {}...\n'.format(trial.number + 1))
    args = setup_args()
    unset_random_seed()
    # override args with optuna suggessions
    # override number of epoch, number of labled, out folder
    args.epochs = 25
    args.n_labeled = 250
    args.out = args.out + 'optuna_search_test'

    # region override args of interest with hyper params to tune from optuna suggest
    args.T_softmax = trial.suggest_int('T_softmax', 1, 100)
    args.num_labeled_per_class = trial.suggest_int('num_labeled_per_class', 1, 3)
    args.alpha = trial.suggest_float('alpha', 0.1, 0.9)
    args.lambda_u = trial.suggest_int('lambda_u', 1, 105)
    args.lr = trial.suggest_float('lr', 1e-5, 0.1)
    args.T = trial.suggest_float('T', 0.001, 1)
    args.K = trial.suggest_int('K', 2, 4)
    args.ema_decay = 0.99
    args.scaling_loss = 1.0
    args.max_iter_gtg = trial.suggest_int('max_iter_gtg', 1, 5)

    # best_val_acc, _, _ = main(logEnabled=False, args=args, isOptuna=True)
    epoch = args.start_epoch
    best_val_acc = 0
    for output in main(logEnabled=False, args=args, isOptuna=True):  # Please Note: Uncomment the yield statement in main() method of train_gl_jsd_without_mixup before running this script
        best_val_acc, _, _ = output
        # report for pruning
        trial.report(best_val_acc, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
        epoch += 1
    evaluation_score = best_val_acc

    return evaluation_score


if __name__ == '__main__':
    # console_log_file_name = 'Optuna_console_Log_{}.txt'.format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    # sys.stdout = open(console_log_file_name, 'a')
    study = optuna.create_study(direction='maximize',
                                study_name='optuna_gl_jsd_wm_exp1',
                                storage='sqlite:///optuna_gl_jsd_wm_exp1.db',
                                load_if_exists=True)
    study.optimize(objective,
                   n_trials=60)  # if runningg in P paralel proceses, the set n_trials = (intended number of trials) / P
    time.sleep(1)
    print('\nBest objective value: {}'.format(study.best_value))
    print('\nBest trial: {}'.format(str(study.best_trial)))
    print('\n\nBest parameter value: {}'.format(str(study.best_params)))


