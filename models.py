from __future__ import print_function
from layers import *
from metrics import *
from inits import glorot
import sys
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class HAMC(Model):
    def __init__(self, placeholders, input_dim, hidden_sizes, support, n_heads=4, **kwargs):
        super(HAMC, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.pred = 0.0
        self.n_heads = n_heads

        self.num_motifs = len(support)
        motif_positions = np.zeros(self.num_motifs)
        for i in range(self.num_motifs):
            motif_positions[i] = support[i][2][0]
        self.motif_positions = np.array(motif_positions).astype('int32')

        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # Weight decay for fc layer
        for var in self.layers[-1].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # softmax cross-entropy
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                      self.placeholders['labels_mask'])
        self.predict()

        self.loss += tf.cast(tf.reduce_sum(self.pred) * 0, tf.float32)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        # For each motif convolution layer
        # Input: N * C tensor, sparse tensor for first layer
        # Output: M * N * H tensor
        n_layers = len(self.hidden_sizes)
        self.inputs = tf.sparse_transpose(self.inputs)
        for i in range(n_layers):
            input_dim = self.input_dim if i == 0 else self.hidden_sizes[i - 1]
            sparse_inputs = (i == 0)
            self.layers.append(AttentionConvolution(name='Conv_l' + str(i),
                                                        input_dim=input_dim,
                                                        output_dim=self.hidden_sizes[i],
                                                        placeholders=self.placeholders,
                                                        motif_positions=self.motif_positions,
                                                        dropout=True,
                                                        bias=True,
                                                        sparse_inputs=sparse_inputs,
                                                        n_heads=self.n_heads,
                                                        is_init=(i == 0),
                                                        is_final=(i == n_layers - 1),
                                                        logging=self.logging))
            if i < n_layers - 1:
                # For each motif attention layer
                # Input: M * N * H tensor
                # Output: N * H tensor
                self.layers.append(MCAttention(name='Attn_l' + str(i),
                                               hidden_size=self.hidden_sizes[i], n_heads=self.n_heads))
                # self.layers.append(Concat(name='Concat_l' + str(i)))
            else:
                self.layers.append(Concat_forMGAN(name='Concat_l' + str(i)))
                # self.layers.append(MCAttention_Final_Ver(name='Attn_l' + str(i),
                #                                   hidden_size=self.hidden_sizes[i], n_heads=self.n_heads, is_final=True))
        # For fully connected layer
        # Input: (M * N) * H tensor
        # Output: N * D tensor
        self.layers.append(Dense(name='FC',
                                 input_dim=self.hidden_sizes[-1] * self.num_motifs * self.n_heads,
                                 # input_dim=self.hidden_sizes[-1] * self.n_heads,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 bias=True,
                                 logging=self.logging))

    def predict(self):
        act = tf.nn.softmax(self.outputs)
        self.pred = tf.one_hot(tf.argmax(act, 1), self.output_dim, on_value=1, off_value=0)
        return self.pred

    def _predict(self):
        return self.pred


