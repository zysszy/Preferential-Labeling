from copy import deepcopy
from .network import sat_infer, sat_loss, sat_semi_loss, sat_loss_each_variable

import multiprocessing
import random
import time
import sys
import os
import struct
import gzip
import pickle
import tqdm

from copy import deepcopy
import numpy as np
import tensorflow as tf

randomtimes = 20

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class SATModel:
    def __init__(self, config):
        self.config = config

        self.pre = 0
        print(bcolors.OKBLUE + "Build network . . ." + bcolors.ENDC)
        self._build_network()
        self.global_step = tf.get_variable("global_step", initializer=tf.constant_initializer(0, dtype=tf.int32), shape=(), trainable=False)
        self.increase_global_step = tf.assign_add(self.global_step, 1)
        print(bcolors.OKBLUE + "Network built!" + bcolors.ENDC)

        self.saver = tf.train.Saver(tf.trainable_variables() + [self.global_step], max_to_keep=1000000)
        tfconfig = tf.ConfigProto()
        tfconfig.allow_soft_placement = True
        tfconfig.gpu_options.allow_growth = True  # pylint: disable=E1101
        self.sess = tf.Session(config=tfconfig)
        self.sess.run(tf.global_variables_initializer())
        self.train_data = iter(self._fetch_train_data())
        self.eval_data = iter(self._fetch_eval_data())

    def _fetch_train_data(self):
        while True:
            files = os.listdir(self.config.train_data)
            random.shuffle(files)
            for f in files:
                if f <= "000000c7.gz":
                    continue
                try:
                    finput = gzip.open(os.path.join(self.config.train_data, f))

                    num_vars, num_clauses, labels, edges, lits_index, clauses_index = pickle.load(finput)
                    yield {
                        "num_vars": num_vars,
                        "num_clauses": num_clauses,
                        "labels": labels,
                        "edges": edges,
                        "lits_index": lits_index,
                        "clauses_index": clauses_index,
                        "oriclauses_index": deepcopy(clauses_index),
                    }
                except Exception as e:
                    print(f"Load ERROR {f}: {e}")

    def _fetch_eval_data(self):
        while True:
            files = os.listdir(self.config.eval_data)
            for f in files:
                try:
                    finput = gzip.open(os.path.join(self.config.eval_data, f))
                    num_vars, num_clauses, labels, edges, lits_index, clauses_index = pickle.load(finput)
                    yield {
                        "num_vars": num_vars,
                        "num_clauses": num_clauses,
                        "labels": labels,
                        "edges": edges,
                        "lits_index": lits_index,
                        "clauses_index": clauses_index,
                        "oriclauses_index": deepcopy(clauses_index),
                    }
                except Exception as e:
                    print(f"Load ERROR {f}: {e}")

    def _build_network(self):
        with tf.name_scope("train_data"):
            self.oriclauses_index = tf.placeholder(tf.float32, shape=[None], name="oriclauses_index")
            num_vars = self.input_num_vars = tf.placeholder(tf.int64, shape=[], name="input_num_vars")
            num_clauses = self.input_num_clauses = tf.placeholder(tf.int64, shape=[], name="input_num_clauses")
            labels = self.input_labels = tf.placeholder(tf.int64, shape=[None], name="input_labels")
            edges = self.input_edges = tf.placeholder(tf.int64, shape=[None, 2], name="input_edges")

            lits_index = self.lits_index = tf.placeholder(tf.int64, shape=[None], name="lits_index")
            clauses_index = self.clauses_index = tf.placeholder(tf.int64, shape=[None], name="clauses_index")

            edges = tf.SparseTensor(indices=edges, values=tf.ones(shape=[tf.shape(edges)[0]]), dense_shape=[num_clauses, num_vars * 2])
            edges = tf.sparse.reorder(edges)
            sum0 = tf.math.sqrt(tf.sparse.reduce_sum(edges, 0, keepdims=True)) + 1e-6
            sum1 = tf.math.sqrt(tf.sparse.reduce_sum(edges, 1, keepdims=True)) + 1e-6
            edges = edges / sum0 / sum1
        with tf.name_scope("train_infer"):
            self.dropout_rate = tf.get_variable("dropout_rate", initializer=tf.constant_initializer(0, dtype=tf.float32), shape=(), trainable=False)
            self.infer, self.infer_semi = infer, infer_semi = sat_infer(num_vars, num_clauses, edges, self.dropout_rate, lits_index, clauses_index)
            self.softmax_infer = tf.nn.softmax(infer, -1)
            self.softmax_infer_semi = tf.nn.softmax(infer_semi, -1)

        with tf.name_scope("train_loss"):
            self.loss = loss = sat_loss(infer, labels)
        with tf.name_scope("train_loss_each_variable"):
            self.loss_each_variable = sat_loss_each_variable(infer, labels)
        with tf.name_scope("semi_loss_0"):
            self.semi_loss_0 = semi_loss_0 = sat_semi_loss(infer_semi, 0)
        with tf.name_scope("semi_loss_1"):
            self.semi_loss_1 = semi_loss_1 = sat_semi_loss(infer_semi, 1)

        if not self.config.is_evaluate:
            with tf.name_scope("train_optimize"):
                optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

                with tf.name_scope("train_loss"):
                    grads = optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
                    grads = [(tf.clip_by_norm(grad, self.config.clip_norm) if grad is not None else None, var) for grad, var in grads]
                    self.optimize = optimizer.apply_gradients(grads)

                with tf.name_scope("train_semi_loss_0"):
                    grads = optimizer.compute_gradients(semi_loss_0, colocate_gradients_with_ops=True)
                    grads = [(tf.clip_by_norm(grad, self.config.clip_norm) if grad is not None else None, var) for grad, var in grads]
                    self.optimize_semi_0 = optimizer.apply_gradients(grads)

                with tf.name_scope("train_semi_loss_1"):
                    grads = optimizer.compute_gradients(semi_loss_1, colocate_gradients_with_ops=True)
                    grads = [(tf.clip_by_norm(grad, self.config.clip_norm) if grad is not None else None, var) for grad, var in grads]
                    self.optimize_semi_1 = optimizer.apply_gradients(grads)

    def dic2dict(self, dic):
        return {
            self.input_num_vars: dic["num_vars"],
            self.input_num_clauses: dic["num_clauses"],
            self.input_labels: dic["labels"],
            self.input_edges: dic["edges"],
            self.lits_index: dic["lits_index"],
            self.clauses_index: dic["clauses_index"],
            self.oriclauses_index: dic["oriclauses_index"],
        }

    def train(self, steps):
        global randomtimes
        while steps > 0:
            self.change_dropout(0.1)
            tqdm_data = tqdm.tqdm(list(range(self.config.train_length)))
            tqdm_data.desc = "Train         :"
            sum_loss = 0
            sum_corr = 0
            num_data = 0
            for _ in tqdm_data:
                dic = next(self.train_data)

                lits_index = self.get_ranges(dic["lits_index"])
                clauses_index = self.get_ranges(dic["clauses_index"])

                min_losses = [1e9] * len(lits_index)
                max_losses = [0] * len(lits_index)
                min_lits_index = np.copy(dic["lits_index"])
                min_clauses_index = np.copy(dic["clauses_index"])
                max_lits_index = np.copy(dic["lits_index"])
                max_clauses_index = np.copy(dic["clauses_index"])

                for t in range(randomtimes):
                    dd = dict(dic.items())
                    dd["lits_index"] = self.random_shuffle(dd["lits_index"])
                    dd["clauses_index"] = self.random_shuffle(dd["clauses_index"])
                    loss_each_variable = self.sess.run(self.loss_each_variable, feed_dict=self.dic2dict(dd))

                    for i, ((lits_begin, lits_end), (clause_begin, clause_end)) in enumerate(zip(lits_index, clauses_index)):
                        cur_loss = np.average(loss_each_variable[lits_begin:lits_end])
                        cur_loss = float(cur_loss)
                        if cur_loss < min_losses[i]:
                            min_losses[i] = cur_loss
                            min_lits_index[lits_begin:lits_end] = dd["lits_index"][lits_begin:lits_end]
                            min_clauses_index[clause_begin:clause_end] = dd["clauses_index"][clause_begin:clause_end]
                        if cur_loss > max_losses[i]:
                            max_losses[i] = cur_loss
                            max_lits_index[lits_begin:lits_end] = dd["lits_index"][lits_begin:lits_end]
                            max_clauses_index[clause_begin:clause_end] = dd["clauses_index"][clause_begin:clause_end]

                mindd = dict(dic.items())
                mindd["lits_index"] = min_lits_index
                mindd["clauses_index"] = min_clauses_index
                maxdd = dict(dic.items())
                maxdd["lits_index"] = max_lits_index
                maxdd["clauses_index"] = max_clauses_index

                _, loss, infer, _, step = self.sess.run([self.optimize, self.loss, self.softmax_infer, self.increase_global_step, self.global_step], feed_dict=self.dic2dict(mindd))
#                _, loss_semi_1 = self.sess.run([self.optimize_semi_1, self.semi_loss_1], feed_dict=self.dic2dict(mindd))
#                _, loss_semi_0 = self.sess.run([self.optimize_semi_0, self.semi_loss_0], feed_dict=self.dic2dict(maxdd))

                corr = np.sum(np.argmax(infer, -1) == dic["labels"]) / len(infer)
                loss = loss#(loss + (loss_semi_0 + loss_semi_1) / 2) / 2
                sum_corr += corr
                sum_loss += loss
                num_data += 1

                tqdm_data.desc = f"Train L:{loss:08.06f} P:{corr:08.06f} AL:{sum_loss/num_data:08.06f} AP:{sum_corr/num_data:08.06f}"
                steps -= 1
                if steps <= 0:
                    break
            print (self.sess.run(self.global_step))
            self.evaluate(self.config.eval_data_size)
            self.saver.save(self.sess, os.path.join(self.config.logdir, "debug"), global_step=self.global_step)

    def get_ranges(self, array):
        last_end = 0
        result = []
        for i, idx in enumerate(array):
            if i != 0 and idx == 0:
                result.append((last_end, i))
                last_end = i
        result.append((last_end, len(array)))
        return result

    def random_shuffle(self, array):
        last_end = 0
        array = np.copy(array)
        for i, idx in enumerate(array):
            if i != 0 and idx == 0:
                np.random.shuffle(array[last_end:i])
                last_end = i
        np.random.shuffle(array[last_end:])
        return array

    def gen_eval_set(self, eval_data):
        eval_data = dict(eval_data.items())
        eval_data[self.lits_index] = self.random_shuffle(eval_data[self.lits_index])
        eval_data[self.clauses_index] = self.random_shuffle(eval_data[self.clauses_index])
        return eval_data

    def checkSAT(self, infer, data):
        edges = data["edges"]
        dic = {}
        for key in edges:
            if key[0] not in dic:
                dic[key[0]] = []
            dic[key[0]].append(key[1])

        index = data["oriclauses_index"]
        l = []
        for i in range(len(index) - 1):
            if index[i] > index[i + 1]:
                l.append(index[i] + 1)
        l.append(index[-1] + 1)
        num_clauses = l
        startIndex = 0

        ret = []
        for i in range(len(num_clauses)):
            SAT = True
            for t in range(num_clauses[i]):
                now = startIndex + t
                if now in dic:
                    l = dic[now]
                    cla = False
                    for k in l:
                        tar = False
                        sub = 0
                        if k >= data["num_vars"]:
                            sub = data["num_vars"]
                        assert infer[k - sub] == 1 or infer[k - sub] == 0
                        if infer[k - sub] == 1:
                            tar = True
                        else:
                            tar = False
                        if k >= data["num_vars"]:
                            tar = not tar
                        cla = cla or tar
                    SAT = SAT and cla
                    if SAT == False:
                        break

            startIndex += num_clauses[i]
            if SAT == True:
                ret.append(1)
            else:
                ret.append(0)

        return ret

    def evaluate(self, steps, NUM_TREES=5):
        self.change_dropout(0)

        max_infer_semi = []#[0] * len(lits_index)
        max_infer = []#np.zeros_like(dic["labels"])
        
        for tr in range(NUM_TREES):
            correct = 0
            total = 0
            sat_clauses = 0
            total_clauses = 0
            count = -1
            for step in tqdm.tqdm(list(range(steps))):
                count += 1
                eval_data = next(self.eval_data)
                dic = eval_data
                lits_index = self.get_ranges(dic["lits_index"])
                clauses_index = self.get_ranges(dic["clauses_index"])
                if len(max_infer_semi) <= count:
                    max_infer_semi.append([0] * len(lits_index))
                    max_infer.append(np.zeros_like(dic["labels"]))

                #eval_data = next(self.eval_data)
                labels = dic["labels"]
                num_nodes = int(dic["num_vars"])
                #dic = eval_data

                dd = dict(dic.items())
                dd["lits_index"] = self.random_shuffle(dd["lits_index"])
                dd["clauses_index"] = self.random_shuffle(dd["clauses_index"])

                infer, infer_semi = self.sess.run([self.softmax_infer, self.softmax_infer_semi], feed_dict=self.dic2dict(dd))
                infer_semi = infer_semi[:,1]
                #infer = np.argmax(infer, -1)
                
                for i, ((lits_begin, lits_end), (clause_begin, clause_end)) in enumerate(zip(lits_index, clauses_index)):
                    cur_semi = np.prod(np.max(infer[lits_begin:lits_end], -1))
                    cur_semi = float(cur_semi)
                    if cur_semi > max_infer_semi[count][i]:
                        max_infer_semi[count][i] = cur_semi
                        max_infer[count][lits_begin:lits_end] = np.argmax(infer[lits_begin:lits_end], -1)

                sat_list = self.checkSAT(max_infer[count], eval_data)
                total_clauses += len(sat_list)
                sat_clauses += sum(sat_list)
                correct += int(np.sum(max_infer[count] == labels))
                total += int(num_nodes)

            if self.config.is_evaluate:
                #print (max_infer_semi)
                print(f"percision: {correct / total}")
                print(f"sat: {sat_clauses / total_clauses}")

    def load_model(self, file):
        self.saver.restore(self.sess, file)

    def change_dropout(self, val):
        self.sess.run(tf.assign(self.dropout_rate, val))

    def run_predict(self, num_vars, num_clauses, edges):
        self.change_dropout(0)
        infer = self.sess.run(self.infer, feed_dict={self.input_num_vars: num_vars, self.input_num_clauses: num_clauses, self.input_edges: edges})
        infer = np.argmax(infer, -1).astype(np.int32)
        return infer
