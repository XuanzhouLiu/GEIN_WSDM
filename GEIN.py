
from models.base_model import Model
from common.utils import *
import numpy as np
from settings import *
import pickle as pkl

class GEIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True, use_mid=True
                 , maxlen=20, n_graph_prod=3, mask_ratio=0.2, drop_ratio=0.0, sub_ratio=0.2, use_ssl=True, use_projector_head=False, c2c_graph=3, ssl_weight=1, batch_size = 128, num_communities=4, mid_factor=False):
        super(GEIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                             ATTENTION_SIZE,
                                             use_negsampling, maxlen=maxlen)
        self.debug = {}
        self.batch_size = batch_size
        self.use_projector_head = use_projector_head
        self.use_ssl = use_ssl

        if not use_mid:
            self.cat_embeddings_var = tf.get_variable("cat_embedding_var2", [n_cat, 2*EMBEDDING_DIM])

        with tf.name_scope("Input"):
            self.graph_mid_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen*n_graph_prod])
            self.graph_cat_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen*n_graph_prod])
            self.c2c_graph_cat_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen*c2c_graph])
            #self.i2i2c_graph_cat_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen*c2c_graph])

            self.pos_his_batch_ph = tf.placeholder(tf.int32, [None, maxlen])
            self.position_his = tf.expand_dims(tf.range(maxlen), 0)
            self.position_his = tf.tile(self.position_his, [tf.shape(self.item_his_eb)[0], 1]) #[B, L]
            self.position_his = tf.expand_dims(self.seq_len_ph, 1) - self.position_his
            self.position_his = tf.where(self.position_his>=0, self.position_his, tf.zeros_like(self.position_his))

            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, maxlen],
                                                        name='noclk_mid_batch_ph') 
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, maxlen], name='noclk_cat_batch_ph')

            #NEW
            self.c2c_1hop_ph = tf.placeholder(tf.float32, [None, maxlen, maxlen])
            self.c2c_2hop_ph = tf.placeholder(tf.float32, [None, maxlen, maxlen])
            self.i2i_his_sum_ph = tf.placeholder(tf.float32, [None, maxlen])

            self.c2c_target_1hop_ph = tf.placeholder(tf.float32, [None, maxlen])
            self.c2c_target_2hop_ph = tf.placeholder(tf.float32, [None, maxlen])
            self.i2i2c_target_ph = tf.placeholder(tf.float32, [None, maxlen, 2])
            self.i2i_target_sum_ph = tf.placeholder(tf.float32, [None])

            self.factors_ph = tf.placeholder(tf.int32, [None, maxlen])
            self.factors_idx_ph = tf.placeholder(tf.int32, [None, maxlen])
            self.cliques_ph = tf.placeholder(tf.int32, [None, maxlen])
            self.cliques_idx_ph = tf.placeholder(tf.int32, [None, maxlen])

            self.tgt_graph_mid_ph = tf.placeholder(tf.int32, [None, n_graph_prod])
            self.tgt_graph_cat_ph = tf.placeholder(tf.int32, [None, n_graph_prod])


        with tf.name_scope("Embedding_layer"):
            if not use_mid:
                self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
                self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
                self.item_his_eb = self.cat_his_batch_embedded
                self.item_eb = self.cat_batch_embedded

            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                           self.noclk_mid_batch_ph)
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var,
                                                                           self.noclk_cat_batch_ph)
                if use_mid:
                    self.noclk_item_his_eb = tf.concat(
                        [self.noclk_mid_his_batch_embedded[:, 0, :, :], self.noclk_cat_his_batch_embedded[:, 0, :, :]],
                        -1)
                    self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)
                else:
                    self.noclk_item_his_eb = self.noclk_cat_his_batch_embedded[:, 0, :, :]
                    self.noclk_his_eb = self.noclk_cat_his_batch_embedded
                self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
                self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)
            

            pos_embedding_size = 4
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen+1, pos_embedding_size])
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E

            self.graph_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.graph_mid_his_batch_ph)
            self.graph_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.graph_cat_his_batch_ph)
            self.c2c_graph_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.c2c_graph_cat_his_batch_ph)
            #self.i2i2c_graph_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.i2i2c_graph_cat_his_batch_ph)
            
            if mid_factor:
                self.factors_embedd = tf.nn.embedding_lookup(self.mid_embeddings_var, self.factors_ph)#[N, L, E]
            else:
                self.factors_embedd = tf.nn.embedding_lookup(self.cat_embeddings_var, self.factors_ph)#[N, L, E]
            self.cliques = tf.where(self.cliques_ph>=0, self.cliques_ph, tf.zeros_like(self.cliques_ph))
            self.cliques_embedd = tf.gather(self.mid_his_batch_embedded, self.cliques, batch_dims=1)

            self.tgt_graph_mid_embedd = tf.nn.embedding_lookup(self.mid_embeddings_var, self.tgt_graph_mid_ph)
            self.tgt_graph_cat_embedd = tf.nn.embedding_lookup(self.cat_embeddings_var, self.tgt_graph_cat_ph)

            self.tgt_graph_mid = tf.nn.avg_pool1d(self.tgt_graph_mid_embedd, ksize=n_graph_prod, strides=n_graph_prod, padding="VALID")
            self.tgt_graph_cat = tf.nn.avg_pool1d(self.tgt_graph_cat_embedd, ksize=n_graph_prod, strides=n_graph_prod, padding="VALID")
            self.tgt_graph_embedd = tf.concat([self.tgt_graph_mid[:,0,:], self.tgt_graph_cat[:,0,:]], axis=-1)

            if use_mid:
                self.graph_his_eb = tf.concat([self.graph_mid_his_batch_embedded, self.graph_cat_his_batch_embedded], axis=-1)
            else:
                self.graph_his_eb = self.graph_cat_his_batch_embedded
            
            
        if self.use_ssl:
            self.seq_aug_mask_mask = tf.random.categorical(tf.math.log(tf.tile([[mask_ratio,1-mask_ratio]], [tf.shape(self.item_his_eb)[0], 1])), tf.shape(self.item_his_eb)[1])
            self.seq_aug_mask_mask = tf.tile(tf.expand_dims(self.seq_aug_mask_mask,-1),[1,1,tf.shape(self.item_his_eb)[-1]])
            
            self.seq_aug_sub_mask = tf.random.categorical(tf.math.log(tf.tile([[sub_ratio,1-sub_ratio]], [tf.shape(self.item_his_eb)[0], 1])), tf.shape(self.item_his_eb)[1])
            self.seq_aug_sub_mask = tf.tile(tf.expand_dims(self.seq_aug_sub_mask,-1),[1,1,tf.shape(self.item_his_eb)[-1]])
            
            tmp_id = tf.reshape(tf.range(maxlen), [1, -1]) * n_graph_prod
            self.seq_i2i_sub_idx = tf.random.categorical(tf.math.log(tf.tile([[1/n_graph_prod]*n_graph_prod], [tf.shape(self.item_his_eb)[0], 1])), tf.shape(self.item_his_eb)[1])
            self.seq_i2i_sub_idx = tf.cast(tmp_id, tf.int64)+self.seq_i2i_sub_idx
            self.seq_i2i_sub_eb = tf.gather(self.graph_his_eb, self.seq_i2i_sub_idx, batch_dims=1) 
            #self.debug["graph_i2i_eb"] = self.graph_his_eb

            self.item_his_eb_aug = tf.cast(self.seq_aug_mask_mask, tf.float32) * self.item_his_eb
            self.item_his_eb_aug = tf.where(tf.cast(self.seq_aug_sub_mask, bool), self.item_his_eb_aug, self.seq_i2i_sub_eb)
            self.item_his_eb_aug = tf.layers.dropout(self.item_his_eb_aug, drop_ratio, training=True)

        self.graph_his_eb = tf.nn.avg_pool1d(self.graph_his_eb, ksize=n_graph_prod, strides=n_graph_prod, padding="VALID")

        with tf.variable_scope("target_distance_attention", reuse = tf.AUTO_REUSE):
            #self.target_res_c2c_eb = tf.where(tf.tile(tf.expand_dims(self.c2c_target_1hop_ph>0, -1), tf.stack([1, 1, tf.shape(self.item_his_eb)[-1]])), self.item_his_eb, tf.zeros_like(self.item_his_eb))
            self.target_distance = tf.stack([self.c2c_target_1hop_ph, self.c2c_target_2hop_ph], axis=-1)
            self.target_distance = tf.concat([self.target_distance, self.i2i2c_target_ph], axis=-1)
            self.target_distance = tf.concat([self.target_distance, self.position_his_eb], axis=-1)
            self.target_distance = tf.compat.v1.layers.dense(self.target_distance, EMBEDDING_DIM, activation=tf.nn.relu)
            self.target_distance = tf.compat.v1.layers.dense(self.target_distance, 1)

            self.target_res_c2c_eb = self.item_his_eb * self.target_distance
            self.target_edge_eb = din_attention(self.item_eb, self.target_res_c2c_eb, ATTENTION_SIZE, self.mask)
            self.target_edge_eb = self.target_edge_eb[:,0,:]


        with tf.variable_scope("his_distance_attention", reuse = tf.AUTO_REUSE):
            self.i2i_distance = tf.compat.v1.layers.dense(tf.stack([self.c2c_1hop_ph, self.c2c_2hop_ph], axis=-1),
                                                           EMBEDDING_DIM, activation=tf.nn.relu)
            self.i2i_distance = tf.concat(tf.split(tf.compat.v1.layers.dense(self.i2i_distance, 4), 4, axis=-1), axis=0)
            self.i2i_distance = tf.reduce_sum(tf.compat.v1.layers.dense(self.i2i_distance, 1), axis=-1)
            #self.i2i_distance = None
            

        
        with tf.variable_scope("multi_head_attention", reuse = tf.AUTO_REUSE):
            self.k_mask = tf.expand_dims(tf.range(maxlen), axis=0) <= tf.expand_dims(tf.range(maxlen), axis=1)
            self.k_mask = tf.tile(tf.expand_dims(self.k_mask, axis=0), [tf.shape(self.item_his_eb)[0], 1, 1])
            multihead_attention_outputs, self.att = self.graphormer_layer(self.item_his_eb, self.item_his_eb, 1, EMBEDDING_DIM,
                                                               q_mask=tf.ones_like(self.cat_his_batch_ph), k_mask=self.k_mask,
                                                               add=self.i2i_distance)
            #multihead_attention_outputs = self.multihead_self_attention_1(self.item_his_eb, EMBEDDING_DIM)

            if self.use_ssl:
                multihead_attention_outputs_aug, self.att_aug = self.graphormer_layer(self.item_his_eb_aug, self.item_his_eb_aug, 1, EMBEDDING_DIM,
                                                                        q_mask=tf.ones_like(self.cat_his_batch_ph), k_mask=self.k_mask,
                                                                        add=self.i2i_distance)
                #multihead_attention_outputs_aug = self.multihead_self_attention_1(self.item_his_eb_aug, EMBEDDING_DIM)
        
        aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                         self.noclk_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="gru")

        if self.use_ssl:
            aux_loss_2 = self.auxiliary_loss(multihead_attention_outputs_aug[:, :-1, :], self.item_his_eb[:, 1:, :],
                                    self.noclk_item_his_eb[:, 1:, :],
                                    self.mask[:, 1:], stag="gru")
            self.aux_loss = (aux_loss_1+aux_loss_2)/2
        else:
            self.aux_loss = aux_loss_1

        if self.use_ssl:
            self.multihead_attention_outputs_norm = tf.nn.l2_normalize(multihead_attention_outputs, axis= -1)
            self.multihead_attention_outputs_norm_aug = tf.nn.l2_normalize(multihead_attention_outputs_aug, axis= -1)

            self.ssl_loss1 = self.inter_ssl(self.multihead_attention_outputs_norm, self.multihead_attention_outputs_norm_aug)

            self.aux_loss = self.ssl_loss1*ssl_weight + self.aux_loss

        with tf.variable_scope("c2c_attention", reuse = tf.AUTO_REUSE):
            q_mask = tf.ones(tf.shape(self.cat_his_batch_embedded)[:-1])#B x n_c2c*seq_len x E
            k_mask = tf.ones(tf.shape(self.c2c_graph_cat_his_batch_embedded)[:-1])
            self.c2c_attention_outputs, self.c2c_attention_coefs = multihead_attention2(self.cat_his_batch_embedded, self.c2c_graph_cat_his_batch_embedded, EMBEDDING_DIM, EMBEDDING_DIM if use_mid else 2*EMBEDDING_DIM, num_heads=2,
                                                                              key_masks=k_mask, query_masks=q_mask)
            #self.c2c_attention_eb = tf.nn.avg_pool1d(self.c2c_attention_outputs, ksize=c2c_graph, strides=c2c_graph, padding="VALID")
            c2c_attention_result, self.att3 = din_attention(self.cat_batch_embedded, self.c2c_attention_outputs, ATTENTION_SIZE // 2, self.mask, return_att=True)
            c2c_attention_result = c2c_attention_result[:,0]

        inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, c2c_attention_result, self.target_edge_eb, self.tgt_graph_embedd], 1)
        
        
        with tf.variable_scope("hidden_interest_attention"):
            hidden_factor_musk = self.factors_idx_ph>0
            hidden_interest_musk = tf.equal(tf.expand_dims(self.factors_idx_ph, -1), tf.expand_dims(self.cliques_idx_ph, -2))
            #hidden_interest_musk = tf.logical_and(hidden_interest_musk, tf.expand_dims(self.cliques_idx_ph>0,-1))
            #self.hidden_interest_embedd, self.hidden_interest_att = din_attention(self.cat_batch_embedded, self.community_embedded, ATTENTION_SIZE // 2, hidden_interest_musk, return_att=True)
            #inp = tf.concat([inp, self.hidden_interest_embedd[:,0,:]], 1)
            self.hidden_interest_embedd, self.hidden_interest_att = multihead_attention2(self.factors_embedd, self.cliques_embedd, EMBEDDING_DIM, EMBEDDING_DIM, num_heads=2, query_masks=hidden_factor_musk, key_masks=hidden_interest_musk)
            self.hidden_interest, self.iterest_att = din_attention(self.cat_batch_embedded, self.hidden_interest_embedd, ATTENTION_SIZE//2, hidden_factor_musk, return_att=True)
            inp = tf.concat([inp, self.hidden_interest[:,0,:]], 1)

        with tf.variable_scope("multi_head_attention2"):
            multihead_attention_outputss, self.att2 = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36,
                                                                   num_heads=4, dropout_rate=0, is_training=True, return_att=True, add=self.i2i_distance)
            multihead_attention_outputs_v2 = tf.reduce_mean(multihead_attention_outputss, axis=0)
            multihead_attention_outputs3 = tf.compat.v1.layers.dense(multihead_attention_outputs_v2,
                                                                        EMBEDDING_DIM * 4, activation=tf.nn.relu)
            multihead_attention_outputs3 = tf.compat.v1.layers.dense(multihead_attention_outputs3,
                                                                        EMBEDDING_DIM * 2)
            multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
            # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
            print('multihead_attention_outputs_v2.get_shape()', multihead_attention_outputs_v2.get_shape())
        
        with tf.variable_scope('Attention_layer' + str("i")):
            print('self.position_his_eb.get_shape()', self.position_his_eb.get_shape())
            print('self.item_eb.get_shape()', self.item_eb.get_shape())
            attention_output, self.attention_score, self.attention_scores_no_softmax = din_attention_new(self.item_eb,
                                                                                                multihead_attention_outputs_v2,
                                                                                                self.position_his_eb,
                                                                                                ATTENTION_SIZE,
                                                                                                self.mask,
                                                                                                stag=str("i"))
            print('attention_output.get_shape()', attention_output.get_shape())
            att_fea = tf.reduce_sum(attention_output, 1)
            inp = tf.concat([inp, att_fea], 1)

            with tf.name_scope('Attention_layer'):
                with tf.variable_scope("graph_attention"):
                    graph_attention_output = din_attention(self.item_eb, self.graph_his_eb,ATTENTION_SIZE // 2, self.mask)
                graph_att_fea = tf.reduce_sum(graph_attention_output, 1)
                inp = tf.concat([inp, graph_att_fea], 1)


        self.build_fcn_net(inp, use_dice=True)

    def multihead_self_attention_1(self, his_eb, EMBEDDING_DIM):
        multihead_attention_outputs = self_multi_head_attn(his_eb, num_units=EMBEDDING_DIM * 2,
                                                            num_heads=4, dropout_rate=0, is_training=True)

        return multihead_attention_outputs

    def graphormer_layer(self, quiry_embedd, key_embedd, graphormer_lyrs, EMBEDDING_DIM, q_mask, k_mask, add=None, reuse=True):
        if not hasattr(self, "i2i_graphormer_outputs"):
            self.i2i_graphormer_outputs = []
            self.i2i_graphormer_coefs = []
        for i in range(graphormer_lyrs):
            i2i_graphormer_output, i2i_graphormer_coef = multihead_attention2(quiry_embedd, key_embedd, EMBEDDING_DIM * 2, EMBEDDING_DIM * 2, activation_fn=None, num_heads=4, key_masks=k_mask, query_masks=q_mask,
                                                                scope="graphormer_attention_layer_".format(i), reuse=tf.AUTO_REUSE, res_connect=True, final_norm=True, add=add)
            shape = tf.shape(i2i_graphormer_output)

            i2i_graphormer_output = tf.reshape(i2i_graphormer_output, [-1, i2i_graphormer_output.get_shape()[-1]])

            i2i_graphormer_residual = tf.compat.v1.layers.dense(i2i_graphormer_output, EMBEDDING_DIM * 4,
                                                            activation=tf.nn.relu, name="graphormer_residual_1_layer_".format(i), reuse=tf.AUTO_REUSE)
            i2i_graphormer_residual = tf.compat.v1.layers.dense(i2i_graphormer_residual, EMBEDDING_DIM * 2,
                                                                name="graphormer_residual_2_layer_".format(i), reuse=tf.AUTO_REUSE)
            i2i_graphormer_output = tf.reshape(i2i_graphormer_output + i2i_graphormer_residual, shape)

            self.i2i_graphormer_outputs.append(i2i_graphormer_output)
            self.i2i_graphormer_coefs.append(i2i_graphormer_coef)

        return i2i_graphormer_output, i2i_graphormer_coef


    def projector_head(self, in_, stag='projector_head'):
        
        with tf.variable_scope('projector_head', reuse=tf.AUTO_REUSE):
            bn0 = tf.layers.batch_normalization(inputs=in_, name='bn0' + stag, reuse=tf.AUTO_REUSE)
            dnn0 = tf.layers.dense(bn0, 1024, activation=None, name='f0' + stag, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), reuse=tf.AUTO_REUSE)
            # bn1 = tf.layers.batch_normalization(inputs=dnn0, name='bn1' + stag, reuse=tf.AUTO_REUSE)
            dnn0 = tf.nn.sigmoid(dnn0)
            dnn1 = tf.layers.dense(dnn0, 512, activation=None, name='f1' + stag, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), reuse=tf.AUTO_REUSE)
            # dnn1 = tf.layers.batch_normalization(inputs=dnn1, name='bn2' + stag, reuse=tf.AUTO_REUSE)
            dnn1 = tf.nn.sigmoid(dnn1)
            dnn2 = tf.layers.dense(dnn1, 128, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
            # dnn0 = tf.layers.dense(bn1, 512, activation=None, name='f0' + stag,  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003), reuse=tf.AUTO_REUSE)
        return dnn2

    def inter_ssl(self, seq_embed, seq_embed_aug, temperature=0.5, stag='projector_head'):

        seq_embed_projector_embed = tf.reshape(seq_embed, [-1, int(seq_embed.get_shape()[1] * seq_embed.get_shape()[2])])
        seq_embed_aug_projector_embed = tf.reshape(seq_embed_aug, [-1, int(seq_embed_aug.get_shape()[1] * seq_embed_aug.get_shape()[2])])
        
        if self.use_projector_head:
            print("Attention! SSL USE Projector!")
            seq_embed_projector_embed = self.projector_head(seq_embed_projector_embed, stag=stag)
            seq_embed_aug_projector_embed = self.projector_head(seq_embed_aug_projector_embed, stag=stag)
        else:
            print("NO Projector!")
            
        seq_projector_embed_concat = tf.concat((seq_embed_projector_embed, seq_embed_aug_projector_embed), axis = 0)

        self.debug['seq_embed_projector_embed'] = seq_embed_projector_embed
        self.debug['seq_embed_aug_projector_embed'] = seq_embed_aug_projector_embed
        self.debug['seq_projector_embed_concat'] =  seq_projector_embed_concat
        
        # cosin = self.cal_cos(seq_projector_embed_concat, seq_projector_embed_concat)
        cosin = tf.matmul(seq_projector_embed_concat, seq_projector_embed_concat, transpose_b=True)
        
        pos_mask, neg_mask = self.mask_correlated_samples()

        self.debug['cosin'] = cosin
        self.debug['pos_mask'] = pos_mask
        self.debug['neg_mask'] = neg_mask
        
        info_nce = self.infocNCE(cosin, pos_mask, neg_mask, temperature)
        
        return info_nce
        
    def mask_correlated_samples(self):
            
        N = 2 * self.batch_size
        pos_mask = np.ones((N, N), dtype=bool)
        neg_mask = np.zeros((N, N), dtype=bool)
        for i in range(self.batch_size):
            
            
            pos_mask[i, i] = False
            pos_mask[i+self.batch_size, i+self.batch_size] = False
            # pos_mask[i, batch_size + i] = False
            # pos_mask[batch_size + i, i] = False
            
            # for data_book 
            pos_mask[i, i+1] = False
            pos_mask[i+self.batch_size, i+self.batch_size-1] = False
            if i + self.batch_size + 1 < N and i % 2 != 0:
                pos_mask[i+self.batch_size, i+self.batch_size+1] = False
            if i - 1 >= 0 and i % 2 == 0:
                pos_mask[i, i-1] = False
            
            neg_mask[i, self.batch_size + i] = True
            neg_mask[self.batch_size + i, i] = True
        pos_mask = tf.constant(pos_mask, dtype = tf.bool)
        pos_mask = pos_mask[:2*tf.shape(self.item_his_eb)[0],:2*tf.shape(self.item_his_eb)[0]]
        return pos_mask, neg_mask
    
    
    def infocNCE(self, cosin, pos_mask, neg_mask, temperature):
        # version 2 infoNCE
        
        cosin = cosin / temperature
        neg_inf = tf.ones_like(cosin) * (-2 ** 32 + 1)
        all_info = tf.where(pos_mask, cosin, neg_inf)
        all_info_softmax = tf.nn.softmax(all_info, axis=-1)
        
        idx = tf.range(tf.shape(self.item_his_eb)[0])
        pos_idx = tf.reshape(tf.concat([idx + tf.shape(self.item_his_eb)[0], idx], axis=-1), [-1, 1])
        
        pos_softmax = tf.batch_gather(all_info_softmax, pos_idx)
       
        info_nce_loss = tf.reduce_mean(-tf.log(pos_softmax))
        
        
        self.debug['pos_idx'] = pos_idx
        self.debug['pos_softmax'] = pos_softmax
        self.debug['all_info'] = all_info
        self.debug['all_info_softmax'] = all_info_softmax
        
        self.debug['info_nce_loss'] = info_nce_loss
        
        return info_nce_loss

    def edge(self, i, j):
        if i in self.c2c_dict and j in self.c2c_dict[i]:
            return -np.log(self.c2c_dict[i][j]+1e-10)
        elif i == j:
            return -np.log(1e-10)
        return 0
    
    def hop_2_edge(self, i, j):
        if i in self.c2c_dict and j in self.c2c_dict:
            true_edges = len(set(self.c2c_dict[i].keys()).intersection(set(self.c2c_dict[j].keys())))
            pred_edges = (len(self.c2c_dict[i])/len(self.c2c_dict) * len(self.c2c_dict[j]))
            return min(max(true_edges/pred_edges - 1, 0), 4)
        return 0
  
    def train(self, sess, inps):

        loss, accuracy, _, aux_loss, y_hat, y = sess.run([self.loss, self.accuracy, self.optimizer, self.aux_loss, self.y_hat, self.target_ph],
                                                   feed_dict={
                                                       self.uid_batch_ph: inps[0],
                                                       self.mid_batch_ph: inps[1],
                                                       self.cat_batch_ph: inps[2],
                                                       self.mid_his_batch_ph: inps[3],
                                                       self.cat_his_batch_ph: inps[4],
                                                       self.pos_his_batch_ph: inps[5],
                                                       self.mask: inps[6],
                                                       self.graph_mid_his_batch_ph: inps[7],
                                                       self.graph_cat_his_batch_ph: inps[8],
                                                       self.c2c_graph_cat_his_batch_ph: inps[9],
                                                       self.target_ph: inps[10],
                                                       self.seq_len_ph: inps[11],
                                                       self.lr: inps[12],
                                                       self.noclk_mid_batch_ph: inps[13],
                                                       self.noclk_cat_batch_ph: inps[14],

                                                       self.c2c_target_1hop_ph:inps[15],
                                                       self.c2c_1hop_ph:inps[16],
                                                       self.c2c_target_2hop_ph:inps[17],
                                                       self.c2c_2hop_ph:inps[18],
                                                       self.i2i_his_sum_ph: inps[19],
                                                       self.i2i_target_sum_ph: inps[20],
                                                       self.i2i2c_target_ph: inps[21],
                                                       self.factors_ph: inps[22],
                                                       self.factors_idx_ph: inps[23],
                                                       self.cliques_ph: inps[24],
                                                       self.cliques_idx_ph: inps[25],
                                                       self.tgt_graph_mid_ph: inps[26],
                                                       self.tgt_graph_cat_ph: inps[27]
                                                   })
        return loss, accuracy, aux_loss
    

    def calculate(self, sess, inps):
        probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict={
                                                       self.uid_batch_ph: inps[0],
                                                       self.mid_batch_ph: inps[1],
                                                       self.cat_batch_ph: inps[2],
                                                       self.mid_his_batch_ph: inps[3],
                                                       self.cat_his_batch_ph: inps[4],
                                                       self.pos_his_batch_ph: inps[5],
                                                       self.mask: inps[6],
                                                       self.graph_mid_his_batch_ph: inps[7],
                                                       self.graph_cat_his_batch_ph: inps[8],
                                                       self.c2c_graph_cat_his_batch_ph: inps[9],
                                                       self.target_ph: inps[10],
                                                       self.seq_len_ph: inps[11],
                                                       self.noclk_mid_batch_ph: inps[12],
                                                       self.noclk_cat_batch_ph: inps[13],

                                                       self.c2c_target_1hop_ph:inps[14],
                                                       self.c2c_1hop_ph:inps[15],
                                                       self.c2c_target_2hop_ph:inps[16],
                                                       self.c2c_2hop_ph:inps[17],
                                                       self.i2i_his_sum_ph: inps[18],
                                                       self.i2i_target_sum_ph: inps[19],
                                                       self.i2i2c_target_ph: inps[20],
                                                       self.factors_ph: inps[21],
                                                       self.factors_idx_ph: inps[22],
                                                       self.cliques_ph: inps[23],
                                                       self.cliques_idx_ph: inps[24],
                                                       self.tgt_graph_mid_ph: inps[25],
                                                       self.tgt_graph_cat_ph: inps[26],
        })
        return probs, loss, accuracy, aux_loss