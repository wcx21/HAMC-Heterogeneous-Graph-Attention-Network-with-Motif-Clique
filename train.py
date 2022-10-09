from __future__ import division
from __future__ import print_function

import time
import optparse
import tensorflow as tf
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *
from models import MotifCNN, HAMC
from metrics import *

# Settings
flags = tf.app.flags

# Set random seed
# r_seed = random.randint(1, 10000)
r_seed = 3641
print(f"random seed = {r_seed}")

flags.DEFINE_integer('seed', r_seed, 'Random seed')
flags.DEFINE_integer('conv_layers', 3, 'Number of convolution-pooling layers.')
flags.DEFINE_integer('attention_type', 0, 'Type of attention, 0 for dot product, 1 for single weight')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_float('learning_rate', 0.003, 'Initial learning rate.')
# flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('dropout_fc', 0.5, 'Dropout rate for FC layer (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_bool('calc_motif', False, 'Calculate motif from scratch')
flags.DEFINE_string('m', '', 'model')
flags.DEFINE_string('d', '', 'ds')
flags.DEFINE_string('l', '', 'lb')
# flags.DEFINE_string('dataset', '', 'model')


def dblp_p_flag():
    flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_string('dataset', 'dblp-p', 'Dataset string.')
    flags.DEFINE_string('motif_def', './motif_def_dblp.json', 'JSON file where motif definitions are given')
    flags.DEFINE_string('motif', 'apv,pap,pp1,pp2', 'Motifs used for model')
    flags.DEFINE_string('hidden_sizes', '128,32,10', 'Hidden layer sizes')


def imdb_flag():
    flags.DEFINE_string('motif_def', './motif_def_imdb.json', 'JSON file where motif definitions are given')
    flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_string('dataset', 'imdb', 'Dataset string.')
    flags.DEFINE_string('motif', 'dma,amd,mam,mdm', 'Motifs used for model')
    flags.DEFINE_string('hidden_sizes', '128,32,3', 'Hidden layer sizes')


parser = optparse.OptionParser()
parser.add_option("-m", '--model', action='store',
                      dest='model', default='HAMC', type='string',
                      help='Choose a model to train, default:HAMC')

parser.add_option('-d', '--dataset', action='store',
                      dest='dataset', default='dblp-p', type='string',
                      help='Choose a dataset to train, default:dblp-p')

parser.add_option('-l', '--label', action='store',
                      dest='label', default=10, type='int',
                      help='proportion of training set default:10%')

options, args = parser.parse_args()

model_dict = {
    'HAMC': HAMC,
    'Meta-GNN': MotifCNN
}

m_clique_num_dict = {
    'dblp-p': 4,
    'imdb': 2,
}

param_set_dict = {
    'dblp-p': dblp_p_flag,
    'imdb': imdb_flag,
}

print(options.model, options.dataset)
param_set_dict[options.dataset]()
chosen_model = model_dict[options.model]

FLAGS = flags.FLAGS
tf.reset_default_graph()
tf.set_random_seed(FLAGS.seed)
motif_types = FLAGS.motif.split(',')
hidden_sizes = [int(x) for x in FLAGS.hidden_sizes.split(',')]
checkpt_file = 'log/{}_{}.ckpt'.format(FLAGS.dataset, options.model)


# Define model evaluation function
def evaluate(sess, model, features, support, labels, mask, placeholders):
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model._predict()], feed_dict=feed_dict_val)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.outputs], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2]


def run_train(model_class, m_clique_num=4):
    # Initialize session
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        load_data(FLAGS.dataset, motif_types, load_ind=True, train_p=options.label, calc_motif=FLAGS.calc_motif, motif_def=FLAGS.motif_def)

    adj = discover_motif_clique(adj, m_clique_num)

    features = preprocess_features(features)

    support = preprocess_adj_zero_one(adj)
    if model_class == MotifCNN:
        support = preprocess_adj(adj)

    num_motifs = len(motif_types)
    channels = features[2][1]

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32, shape=(None, None)),
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_motifs)],
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'dropout_fc': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    # Create model
    # good_adj = preprocess_adj2(adj)

    if model_class == HAMC:
        model = model_class(placeholders, input_dim=channels, hidden_sizes=hidden_sizes, support=support, n_heads=3, logging=True)
    else:
        model = model_class(placeholders, input_dim=channels, hidden_sizes=hidden_sizes, support=support, logging=True)

    saver = tf.train.Saver()
    sess = tf.Session()
    # Init variables
    sess.run(tf.global_variables_initializer())

    # Train model
    cost_val = []
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict_val = construct_feed_dict(features, support, y_val, val_mask, placeholders)

    for epoch in range(FLAGS.epochs):
        # Construct feed dictionary
        # feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['dropout_fc']: FLAGS.dropout_fc})

        # Training
        s = time.time()
        sess.run([model.opt_op], feed_dict=feed_dict)
        e = time.time()

        # feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        outs = sess.run([model.loss, model.accuracy, model._predict()], feed_dict=feed_dict)
        macro_f1_t, micro_f1_t = compute_f1(outs[-1], y_train, train_mask)
        # Validation
        # cost, acc_v, pred_val = evaluate(sess, model, features, support, y_val, val_mask, placeholders)
        cost, acc_v, pred_val = sess.run([model.loss, model.accuracy, model._predict()], feed_dict=feed_dict_val)

        macro_f1_v, micro_f1_v = compute_f1(pred_val, y_val, val_mask)
        cost_val.append(cost)
        # Print results
        # print('Epoch:', '%04d' % (epoch + 1), 'train_pred=', outs[-1].sum(0), 'val_pred=', pred_val.sum(0))
        print('Epoch:', '%04d' % (epoch + 1), 'train_loss=', '{:.5f}'.format(outs[0]),
              'train_micro_f1=', '{:.5f}'.format(micro_f1_t),
              'train_macro_f1=', '{:.5f}'.format(macro_f1_t), 'time=', '{:.5f}'.format(e - s))
        print('val_loss=', '{:.5f}'.format(cost),
              'val_micro_f1=', '{:.5f}'.format(micro_f1_v),
              'val_macro_f1=', '{:.5f}'.format(macro_f1_v))
        if epoch > FLAGS.early_stopping and cost_val[-1] >= np.max(cost_val[-10: -1]):
            # if epoch > FLAGS.early_stopping and (cost_val[-1] + cost_val[-2] + cost_val[-3]) / 3 > np.mean(cost_val[-(FLAGS.early_stopping + 1): -1]):
            #     print(cost_val)
            print('Early stopping...')
            break
        if epoch > 5 and cost_val[-1] <= np.min(cost_val[-5: -1]):
            saver.save(sess, checkpt_file)
        # sess.graph.finalize()
        # to make sure no memory leak
    print('Optimization Finished!')
    saver.restore(sess, checkpt_file)

    cost, acc, pred_test = evaluate(sess, model, features, support, y_test, test_mask, placeholders)
    macro_f1, micro_f1 = compute_f1(pred_test, y_test, test_mask)

    print('Test set results:', 'cost=', '{:.5f}'.format(cost),
          'micro_f1=', '{:.5f}'.format(micro_f1),
          'macro_f1=', '{:.5f}'.format(macro_f1))

    return macro_f1, micro_f1


if __name__ == '__main__':
    m_c_num = m_clique_num_dict[options.dataset] if chosen_model == HAMC else 0
    macro_f1, micro_f1 = run_train(chosen_model, m_c_num)
