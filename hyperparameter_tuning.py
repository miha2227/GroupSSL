import random
from math import log10
import time
import json
from train import args_setup, main

ALLOWED_RANDOM_SEARCH_PARAMS = ['log', 'int', 'float', 'item']


def random_search(random_search_spaces={
    "T_softmax": ([1, 100], "int"),
    "num_labeled_per_class": ([1, 3], "int"),
    #"alpha": ([0.5, 0.99], "float"),
    "lambda_u": ([75, 100], "int"),
    "lr": ([1e-5, 0.1], "float"),
},
        num_search=20, epochs=20,
        patience=5):
    """
    Samples N_SEARCH hyper parameter sets within the provided search spaces
    and returns the best model.
    See the grid search documentation above.
    Additional/different optional arguments:
        - random_search_spaces: similar to grid search but values are of the
        form
        (<list of values>, <mode as specified in ALLOWED_RANDOM_SEARCH_PARAMS>)
        - num_search: number of times we sample in each int/float/log list
    """
    configs = []
    for _ in range(num_search):
        configs.append(random_search_spaces_to_config(random_search_spaces))

    return findBestConfig(configs, epochs)


def findBestConfig(configs, epochs):
    """
    Get a list of hyperparameter configs for random search or grid search,
    trains a model on all configs and returns the one performing best
    on validation set
    """

    best_val = None
    best_config = None
    best_config_id = None
    #best_model = None
    results = []

    args = args_setup()

    args.epochs = epochs
    start = time.time()
    for i in range(len(configs)):
        print("\nEvaluating Config #{} [of {}]:\n".format(
            (i + 1), len(configs)), configs[i])
        args.out = 'random_search_op/Random_Search_{}_epochs_Configuration_{}'.format(epochs, i)
        args.T_softmax = configs[i]['T_softmax']
        args.num_labeled_per_class = configs[i]['num_labeled_per_class']
        #args.alpha = configs[i]['alpha']
        args.lambda_u = configs[i]['lambda_u']
        args.lr = configs[i]['lr']

        b_v_acc, m_v_acc, m_t_acc = main(args=args, use_cuda=True)
        results.append(m_v_acc)

        if not best_val or max(results) > best_val:  # judged based on mean validation accuracy
            best_val, best_config, best_config_id = max(results), configs[i], i+1
    time_taken = time.time() - start
    with open('random_search_log.txt', 'a') as f:
        f.write('Search completed for {} configurations with {} epochs for each'.format(len(configs), epochs))
        f.write('\nTotal tine taken: {} seconds'.format(time_taken))
        f.write('\nBest Config_id: {}'.format(best_config_id))
        f.write('\nBest Configuration Details: \n')
        f.write(json.dumps(best_config))

    print("\nSearch done. Best avg test accuracy = {}".format(best_val))
    print("Best Config_id:", best_config_id)
    print('Best config: ', best_config)
    return best_config, list(zip(configs, results))


def random_search_spaces_to_config(random_search_spaces):
    """"
    Takes search spaces for random search as input; samples accordingly
    from these spaces and returns the sampled hyper-params as a config-object,
    which will be used to construct solver & network
    """

    config = {}

    for key, (rng, mode) in random_search_spaces.items():
        if mode not in ALLOWED_RANDOM_SEARCH_PARAMS:
            print("'{}' is not a valid random sampling mode. "
                  "Ignoring hyper-param '{}'".format(mode, key))
        elif mode == "log":
            if rng[0] <= 0 or rng[-1] <= 0:
                print("Invalid value encountered for logarithmic sampling "
                      "of '{}'. Ignoring this hyper param.".format(key))
                continue
            sample = random.uniform(log10(rng[0]), log10(rng[-1]))
            config[key] = 10 ** (sample)
        elif mode == "int":
            config[key] = random.randint(rng[0], rng[-1])
        elif mode == "float":
            config[key] = random.uniform(rng[0], rng[-1])
        elif mode == "item":
            config[key] = random.choice(rng)

    return config

if __name__ == '__main__':
    random_search()
