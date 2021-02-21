import os, random
import os.path as osp
import numpy as np 
from copy import deepcopy

import sacred
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from model.ingredient import model_ingredient, get_model
from utils import load_data, pickle_save

ex = sacred.Experiment('CIFAR10 Perceptron', ingredients=[model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    epochs       = 12
    bsize        = 200
    lr           = 0.001
    lr_decay     = 0.95
    weight_decay = 0.5
    temp_dir = osp.join('outputs', 'temp')
    seed = 0


@ex.automain
def main(epochs, bsize, lr, lr_decay, weight_decay, temp_dir, seed):
    #############################################################
    # seeding the experiment is a good practice
    np.random.seed(seed)
    random.seed(seed)
    #############################################################
    
    #############################################################
    # create the model
    # and yes you do NOT need to provide any arguments here 
    # (sacred will do that for you)
    model = get_model()
    #############################################################
    
    #############################################################
    # load the data
    X_train, y_train, X_val, y_val, X_test, y_test = \
        load_data(osp.join('data', 'cifar-10-batches-py'))
    #############################################################
    
    #############################################################
    # the best checkpoint
    os.makedirs(temp_dir, exist_ok=True)
    save_name = osp.join(temp_dir, '{}.pt'.\
        format(ex.current_run.config['model']['name']))
    best_val = (0, 
            {'val_acc': float('-inf'), 'train_loss': float('inf')}, 
            model.state_dict())
    #############################################################
    
    for epoch in range(epochs):
        res = model.train_one_epoch(X_train, y_train, X_val, y_val, 
            learning_rate=lr, reg=weight_decay, batch_size=bsize, verbose=True)

        train_loss = np.mean(res['loss_history'])
        val_acc = res['val_acc']

        print('Validation acc [%03d]: %f'%(epoch, val_acc))

        # logging
        ex.log_scalar('val_acc',    val_acc,    step=epoch + 1)
        ex.log_scalar('train_loss', train_loss, step=epoch + 1)

        # learning rate decay 
        lr *= lr_decay
    
        # save the best checkpoint
        if val_acc >= best_val[1]['val_acc']:
            print('New best model in epoch %d.'%epoch)
            best_val = (epoch + 1, res, deepcopy(model.state_dict()))
            pickle_save(save_name, model.state_dict())
    
    #############################################################
    # use the best checkpoint, evaluate on the test set
    model.params = best_val[2]
    test_acc = (model.predict(X_test) == y_test).mean() 
    print('Test acc: %f'%(test_acc))
    #############################################################

    #############################################################
    # wrap up
    ex.info['test_acc'] = test_acc
    ex.add_artifact(save_name)

    return test_acc
