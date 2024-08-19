
import numpy
import time
import os
import random
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import tensorflow as tf
from dataset_process.data_iterator import DataIterator

from models.GEIN import GEIN
from dataset_process.prepare_data import prepare_data
from common.utils import *
from settings import *
import argparse
import pickle as pkl

best_auc = 0.0

def model_selection(model_type, n_uid, n_mid, n_cat, **kwargs):
    print("Model_type: {}".format(model_type))
    model = GEIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, maxlen=kwargs["maxlen"], n_graph_prod=kwargs["num_prod"], mask_ratio = kwargs["mask_ratio"], drop_ratio = kwargs["drop_ratio"], sub_ratio=kwargs["sub_ratio"], use_projector_head=kwargs["use_projector_head"], ssl_weight=kwargs["ssl_weight"], batch_size=kwargs["batch_size"])
    return model


def eval(sess, test_data, model, model_path, model_type='DNN', **kwargs):
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    stored_arr = []
    # for src, tri0_src, tri1_src, tgt in test_data:
    for one_pass_data in test_data:
        nums += 1
        src, tgt, data = one_pass_data
        uids, mids, cats, mid_his, cat_his, pos_his, mid_mask, graph_mid_his, graph_cat_his, c2c_graph_cat_his,\
        target, sl, noclk_mid_his, noclk_cat_his, \
        c2c_target_1hop, c2c_1hop, c2c_target_2hop, c2c_2hop, i2i_his_sum, i2i_target_sum, i2i2c, factors, factors_idx, cliques, cliques_idx, tgt_mid, tgt_cat = prepare_data_c2c(src, tgt, kwargs["maxlen"], return_neg=True, graph_prod_num=kwargs["num_prod"], distance_data=data, causal=True, target_graph=True)
        prob, loss, acc, aux_loss = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, pos_his, mid_mask, graph_mid_his, graph_cat_his, c2c_graph_cat_his, target, sl, noclk_mid_his, noclk_cat_his, c2c_target_1hop, c2c_1hop, c2c_target_2hop, c2c_2hop, i2i_his_sum, i2i_target_sum, i2i2c, factors, factors_idx, cliques, cliques_idx, tgt_mid, tgt_cat])

        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc

    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, model_path)
    return test_auc, loss_sum, accuracy_sum, aux_loss_sum


def train(seed=1234, model_type='DNN', dataset="electronics", batch_size=128, **kwargs):
    train_file = os.path.join(data_path, dataset, "local_train_splitByUser")
    test_file = os.path.join(data_path, dataset, "local_test_splitByUser")
    valid_file = os.path.join(data_path, dataset, "local_valid_splitByUser")
    uid_voc = os.path.join(data_path, dataset, "uid_voc.pkl")
    mid_voc = os.path.join(data_path, dataset, "mid_voc.pkl")
    cat_voc = os.path.join(data_path, dataset, "cat_voc.pkl")
    item_info = os.path.join(data_path, dataset, "item-info")
    reviews_info = os.path.join(data_path, dataset, "reviews-info")

    model_path = "dnn_save_path/ckpt_noshuff" + model_type + str(seed)
    best_model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        t1 = time.time()

        train_distance = os.path.join(data_path, dataset, "train_factor.csv")
        valid_distance = os.path.join(data_path, dataset, "valid_factor.csv")
        test_distance = os.path.join(data_path, dataset, "test_factor.csv")
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                    batch_size, kwargs["maxlen"], shuffle_each_epoch=kwargs["shuffle"], graph_prod_num=kwargs["num_prod"],
                                    distance=train_distance, target=True)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                    batch_size, kwargs["maxlen"], graph_prod_num=kwargs["num_prod"],
                                    distance=test_distance, target=True)
        valid_data = DataIterator(valid_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                    batch_size, kwargs["maxlen"], graph_prod_num=kwargs["num_prod"],
                                    distance=valid_distance, target=True)        

        print('# Load data time (s):', round(time.time() - t1, 2))

        n_uid, n_mid, n_cat = train_data.get_n()
        print('n_uid: {}, n_mid: {}, n_cat: {}'.format(n_uid, n_mid, n_cat))
        t1 = time.time()
        model = model_selection(model_type, n_uid, n_mid, n_cat, batch_size=batch_size, **kwargs)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        print('# Contruct model time (s):', round(time.time() - t1, 2))

        t1 = time.time()
        print('test_auc: %.4f -- test_loss: %.4f -- test_accuracy: %.4f -- test_aux_loss: %.4f' % eval(
            sess, test_data, model, best_model_path, model_type, **kwargs))
        print('# Eval model time (s):', round(time.time() - t1, 2))
        sys.stdout.flush()

        start_time = time.time()
        
        lr = kwargs["lr"]
        print("lr: {}".format(lr))
        valid_flag = False
        for itr in range(kwargs["iters"]):
            iter = 0
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            # for src, tri0_src, tri1_src, tgt in train_data:
            for one_pass_data in train_data:
                src, tgt, data = one_pass_data
                uids, mids, cats, mid_his, cat_his, pos_his, mid_mask, graph_mid_his, graph_cat_his, c2c_graph_cat_his,\
                target, sl, noclk_mid_his, noclk_cat_his, \
                c2c_target_1hop, c2c_1hop, c2c_target_2hop, c2c_2hop, i2i_his_sum, i2i_target_sum, i2i2c, factors, factors_idx, cliques, cliques_idx, tgt_mid, tgt_cat = prepare_data(src, tgt, kwargs["maxlen"], return_neg=True, graph_prod_num=kwargs["num_prod"], distance_data=data, causal=True, target_graph=True)
                loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, pos_his, mid_mask, graph_mid_his, graph_cat_his, c2c_graph_cat_his, target, sl, lr, noclk_mid_his, noclk_cat_his, c2c_target_1hop, c2c_1hop, c2c_target_2hop, c2c_2hop, i2i_his_sum, i2i_target_sum, i2i2c, factors, factors_idx, cliques, cliques_idx, tgt_mid, tgt_cat])

                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iter += 1

                # Print & Save
                sys.stdout.flush()
                if (iter % test_iter) == 0:
                    print('[Time] ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    print('Best_auc:', best_auc)
                    print('itr: %d --> iter: %d --> train_loss: %.4f -- train_accuracy: %.4f -- tran_aux_loss: %.4f' % \
                          (itr, iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                    if iter >= kwargs["valid_batch"] and itr>=kwargs["iters"]-2:
                        valid_flag = True
                    
                    if valid_flag:
                        print('test_auc: %.4f -- test_loss: %.4f -- test_accuracy: %.4f -- test_aux_loss: %.4f' % eval(
                        sess, valid_data, model, best_model_path, model_type, **kwargs))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0


def test(seed=1234, model_type='DNN', dataset = "electronics", batch_size=128, **kwargs):
    train_file = os.path.join(data_path, dataset, "local_train_splitByUser")
    test_file = os.path.join(data_path, dataset, "local_test_splitByUser")
    valid_file = os.path.join(data_path, dataset, "local_valid_splitByUser")
    uid_voc = os.path.join(data_path, dataset, "uid_voc.pkl")
    mid_voc = os.path.join(data_path, dataset, "mid_voc.pkl")
    cat_voc = os.path.join(data_path, dataset, "cat_voc.pkl")
    item_info = os.path.join(data_path, dataset, "item-info")
    reviews_info = os.path.join(data_path, dataset, "reviews-info")


    model_path = "dnn_best_model/ckpt_noshuff" + model_type + str(seed)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        t1 = time.time()
        train_distance = os.path.join(data_path, dataset, "train_factor.csv")
        test_distance = os.path.join(data_path, dataset, "test_factor.csv")
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                    batch_size, kwargs["maxlen"], shuffle_each_epoch=kwargs["shuffle"], graph_prod_num=kwargs["num_prod"],
                                    distance=train_distance, target=True)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, item_info, reviews_info,
                                    batch_size, kwargs["maxlen"], graph_prod_num=kwargs["num_prod"],
                                    distance=test_distance, sort_by_length=True, target=True)

        print('# Load data time (s):', round(time.time() - t1, 2))
        n_uid, n_mid, n_cat = train_data.get_n()

        model = model_selection(model_type, n_uid, n_mid, n_cat, batch_size=batch_size, **kwargs)
        model.restore(sess, model_path)
        print('test_auc: %.4f -- test_loss: %.4f -- test_accuracy: %.4f -- test_aux_loss: %.4f' % eval(
            sess, test_data, model, model_path, model_type, **kwargs))

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--shuffle", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model", type=str, default="SATAI2I")
    parser.add_argument("--dataset", type=str, default="graph_enhanced")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--num_prod", type=int, default=3)
    parser.add_argument("--mask_ratio", type=float, default=0.0)
    parser.add_argument("--drop_ratio", type=float, default=0.0)
    parser.add_argument("--sub_ratio", type=float, default=0.0)
    parser.add_argument("--ssl_weight", type=float, default=0.3)
    parser.add_argument("--use_projector_head", action='store_true', default=False)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--valid_batch", type=int, default=2400)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--maxlen", type=int, default=20)
    parser.add_argument("--aux_weight", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()
    
    SEED = args.seed
    tf.compat.v1.set_random_seed(SEED)
    numpy.random.seed(SEED)
    random.seed(SEED)

    model_type = args.model
    dataset = args.dataset
    num_prod = args.num_prod

    device = args.device
    if device>=0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    if args.train:
        train(model_type=model_type, seed=SEED, maxlen=args.maxlen, dataset=dataset, num_prod=args.num_prod, shuffle = args.shuffle, mask_ratio = args.mask_ratio, drop_ratio = args.drop_ratio, sub_ratio=args.sub_ratio, use_projector_head = args.use_projector_head, ssl_weight=args.ssl_weight, iters=args.iters, valid_batch=args.valid_batch, lr=args.lr, aux_weight=args.aux_weight, batch_size=args.batch_size)
    if args.test:
        test(model_type=model_type, seed=SEED, maxlen=args.maxlen, dataset=dataset, num_prod=args.num_prod, shuffle = args.shuffle, mask_ratio = args.mask_ratio, drop_ratio = args.drop_ratio, sub_ratio=args.sub_ratio, use_projector_head = args.use_projector_head, ssl_weight=args.ssl_weight, lr=args.lr, aux_weight=args.aux_weight, batch_size=args.batch_size)
    
    print('do nothing...')