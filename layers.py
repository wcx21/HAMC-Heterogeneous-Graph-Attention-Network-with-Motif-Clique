import tensorflow as tf

from inits import *

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def dot2(x, y, x_sparse=False, y_sparse=False, sparse=False):
    '''Wrapper for tf.matmul (sparse vs dense).'''
    if sparse or (x_sparse and not y_sparse):
        res = tf.sparse_tensor_dense_matmul(x, y)
    elif y_sparse:
        y_t = tf.sparse_transpose(y)
        x_t = tf.transpose(x)
        res = tf.sparse_tensor_dense_matmul(y_t, x_t)
        res = tf.transpose(res)
    else:
        res = tf.matmul(x, y)
    return res



class Concat_forMGAN(Layer):
    '''Concatenation layer for multiple motifs.'''

    def __init__(self, **kwargs):
        super(Concat_forMGAN, self).__init__(**kwargs)

    def _call(self, inputs):
        # print('Concat inputs shape', inputs.get_shape())

        for i in range(len(inputs)):
            inputs[i] = tf.transpose(inputs[i])

        output = tf.concat(inputs, axis=1)
        return output


class AttentionConvolution(Layer):
    '''
    Motif convolution layer attention version.
    m shrink matrix ver
    '''

    def __init__(self, input_dim, output_dim, motif_positions, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.elu, bias=False, n_heads=1, is_init=False, is_final=False, **kwargs):
        super(AttentionConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.n_heads = n_heads
        self.support = placeholders['support']
        self.is_init = is_init
        self.is_final = is_final

        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.num_motifs = len(motif_positions)
        self.motif_positions = motif_positions  # number of positions for each motif

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        self._input_dim = input_dim

        with tf.variable_scope(self.name + '_vars'):
            for m in range(self.num_motifs):
                for i in range(self.n_heads):
                    if self.is_init:
                        self.vars['shrink_mat_' + str(m) + str(i)] = glorot([output_dim, input_dim], name='shrink_mat_')
                    else:
                        self.vars['shrink_mat_' + str(m) + str(i)] = glorot([output_dim, self.n_heads * input_dim], name='shrink_mat_')
                    for k in range(0, self.motif_positions[m]):
                        # print('roles_num=', self.motif_positions[m])
                        self.vars['att_cal_mat0_' + str(m) + '_' + str(k) + str(i)] = glorot([1, output_dim], name='att_cal_mat_' + str(m) + '_' + str(k))
                        self.vars['att_cal_mat1_' + str(m) + '_' + str(k) + str(i)] = glorot([1, output_dim],
                                                                                            name='att_cal_mat_' + str(
                                                                                                m) + '_' + str(k))

            if self.bias:
                for m in range(self.num_motifs):
                    for i in range(self.n_heads):
                        self.vars['bias_' +
                                  str(m) + str(i)] = zeros([output_dim, 1], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # motif att conv
        new_activations = []
        # for each motif

        for m in range(self.num_motifs):
            act_m = []
            for i in range(self.n_heads):
                x = inputs
                if self.sparse_inputs:
                    x = sparse_dropout(x, 1 - self.dropout,
                                       self.num_features_nonzero)
                else:
                    x = tf.nn.dropout(x, 1 - self.dropout)

                shrink_x = dot2(self.vars['shrink_mat_' + str(m) + str(i)], x, y_sparse=self.sparse_inputs)
                adj_positions = tf.sparse_split(
                    sp_input=self.support[m], num_split=self.motif_positions[m], axis=0)
                # print(adj_positions)
                m_outputs = list()
                # For each position
                for k in range(0, self.motif_positions[m]):

                    m_k_adj = tf.sparse_reduce_sum_sparse(adj_positions[k], axis=0)

                    self._temp_att = [
                        dot2(self.vars['att_cal_mat0_' + str(m) + '_' + str(k) + str(i)], tf.cast(shrink_x, dtype=tf.float32)),
                        dot2(self.vars['att_cal_mat1_' + str(m) + '_' + str(k) + str(i)],
                             tf.cast(shrink_x, dtype=tf.float32))
                    ]

                    wm_0 = m_k_adj.__mul__(self._temp_att[0])
                    wm_1 = m_k_adj.__mul__(tf.transpose(self._temp_att[1]))
                    # wm_0 = self.adj[m].__mul__(self._temp_att[0])
                    # wm_1 = self.adj[m].__mul__(tf.transpose(self._temp_att[1]))
                    weight_mat = tf.sparse_add(wm_0, wm_1)

                    weight_mat = tf.sparse.softmax(weight_mat)
                    # output = tf.transpose(dot2(weight_mat, x_T, y_sparse=self.sparse_inputs))
                    m_k_output = dot2(weight_mat, tf.transpose(shrink_x), x_sparse=True)

                    m_outputs.append(m_k_output)

                i_output = tf.add_n(m_outputs)
                i_output = tf.transpose(i_output)
                if self.bias:
                    # output += self.vars['bias_' + str(m)]
                    i_output += self.vars['bias_' + str(m) + str(i)]
                act_m.append(self.act(i_output))
            # for i in range(self.n_heads):
            #     act_m[i] = self.act(act_m[i])
            m_output = tf.concat(act_m, 0)
            # print(self.n_heads, act_m[0].shape)
            new_activations.append(m_output)

        return new_activations


class MCAttention(Layer):
    '''Attention mechanism for multiple conv unit.'''

    def __init__(self, hidden_size, method='dot_product', num_motifs=1, is_final=False, n_heads=1, **kwargs):
        super(MCAttention, self).__init__(**kwargs)

        self.method = method
        self.is_final = is_final
        self.n_heads = n_heads

        if self.method not in ['dot_product', 'basic']:
            raise NotImplemented('Unsupported attention type')

        with tf.variable_scope(self.name + '_vars'):
            if self.method == 'dot_product':
                # self.vars['attn_w'] = tf.Variable(tf.random_uniform((hidden_size * n_heads, )),
                #                                   name='attn_w')
                # self.vars['bias'] = tf.Variable(tf.random_uniform((num_motifs, )),
                #                                   name='attn_w')
                self.vars['attn_w'] = glorot([hidden_size * n_heads, 1], name='attn_w')
                self.vars['bias'] = glorot([num_motifs, 1], name='bias')
            elif self.method == 'basic':
                self.vars['attn_w'] = tf.Variable(tf.ones((num_motifs,), name="attn_w"))

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        if self.method == 'dot_product':

            for i in range(len(inputs)):
                inputs[i] = tf.transpose(inputs[i])
            attn_act = tf.tensordot(inputs, self.vars['attn_w'], axes=1) + self.vars['bias']
            # attn_act = tf.nn.tanh(attn_act)
            # n_motif * n_nodes

            # attn_act = tf.multiply(attn_act, 1. / tf.norm(self.vars['attn_w']))
            attn_weights = tf.nn.softmax(
                tf.transpose(attn_act))    # n_nodes * n_motif
            # n_hidden * n_nodes * n_motif
            attended = tf.transpose(inputs, [2, 1, 0]) * attn_weights
            # n_motif * n_nodes * n_hidden
            attended = tf.transpose(attended, [2, 1, 0])
            attended = tf.reduce_sum(attended, axis=0)  # n_nodes * n_hidden

            if not self.is_final:
                attended = tf.transpose(attended)  # n_hidden * n_node, because i like that

        elif self.method == 'basic':
            attended = tf.multiply(tf.transpose(inputs, [1, 2, 0]), tf.nn.softmax(self.vars['attn_w']))
            attended = tf.transpose(attended, [2, 0, 1])
            attended = tf.reduce_sum(attended, axis=0)
        return attended


