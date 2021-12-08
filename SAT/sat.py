
import satnetwork
from argparse import ArgumentParser
import os
import tensorflow as tf
import pickle

def main():
    parser = ArgumentParser()
    parser.add_argument('--queue_max_size', type=int, default=128)
    parser.add_argument('--num_threads', type=int, default=48)
    parser.add_argument('--logdir', type=str, default='train')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clip_norm', type=float, default=0.65)
    parser.add_argument('--gpu_list', type=str, default='')

    parser.add_argument('--gen_num_total', type=int, default=5000)
    parser.add_argument('--gen_num_each_min', type=int, default=30)
    parser.add_argument('--gen_num_each_max', type=int, default=50)
    parser.add_argument('--gen_rate_clauses_min', type=float, default=3.5)
    parser.add_argument('--gen_rate_clauses_max', type=float, default=5.5)
    parser.add_argument('--gen_rate_three', type=float, default=1.0)

    parser.add_argument('--train_steps', type=int, default=1000000)

    parser.add_argument('--load_model', type=str)
    parser.add_argument('--is_evaluate', type=bool, default=False)

    parser.add_argument('--dump_model', type=bool, default=False)

    parser.add_argument('--save_data_path', type=str, default=None)

    parser.add_argument('--eval_output', type=str, default='evalresult.pkl')

    parser.add_argument('--train_data', type=str, default=None)
    parser.add_argument('--eval_data', type=str, default=None)

    parser.add_argument('--eval_data_size', type=int, default=20)
    parser.add_argument('--train_length', type=int, default=50000)

    config = parser.parse_args()

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_list

    model = satnetwork.SATModel(config)

    if config.load_model:
        model.load_model(config.load_model)

    if config.dump_model:
        dumps = {variable.name: model.sess.run(variable)for variable in tf.trainable_variables()}
        pickle.dump(dumps, open('modeldump.pkl', 'wb'))
        return

    if config.is_evaluate:
        model.load_model(config.load_model)
        model.evaluate(config.eval_data_size, 101)
    else:
        model.train(config.train_steps)

if __name__ == '__main__':
    main()
