# -*- coding: utf-8 -*-

import math

import numpy as np
import tensorflow as tf
import logging

import joblib


OPTIMIZER_NAME = {"adam": "AdamOptimizer",
                  "sgd": "GradientDescentOptimizer"}


class TripletEmbedding(object):
    """
    Tripet Embedding model
    """

    def __init__(self, n_entity=None, n_relation=None, entity_factors=None, relation_factors=None,
                 hidden_units=None, lambda_w=10e-3, sigma=1.0, activation_func="relu",
                 optimizer="adam", learning_rate=10e-2, w_regularization=10e-4,
                 n_negative_sample=5):
        """
        :param n_entity: the number of vocabrary of entities
        :param n_relation: the number of vocabrary of entities
        :param entity_factors: dimension for entity factors
        :param relation_factors: dimension for relation factors
        :param hidden_units: list of units for hidden layers
        :param lambda_w: the weight of regularization term
        :param sigma: the parameter for sigmoid function
        :param activation_func: the name of activation function
        :param optimizer: the name of optimizer
        :param n_nagative_sample: the number of negative samples per one positive
        """
        if not hasattr(tf.nn, activation_func):
            raise("TensorFlow has no such activation func"
                  .format(activation_func))
        if not hasattr(tf.train, OPTIMIZER_NAME[optimizer]):
            raise("TensorFlow has no such optimizer {}".format(optimizer))

        self.config = config = {}
        config["n_entity"] = n_entity
        config["n_relation"] = n_relation
        config["hidden_units"] = hidden_units
        config["w_regularization"] = w_regularization
        config["sigma"] = sigma
        config["learning_rate"] = learning_rate
        config["entity_factors"] = entity_factors
        config["relation_factors"] = relation_factors
        config["activation_func"] = activation_func
        config["optimizer"] = optimizer
        config["n_negative_sample"] = n_negative_sample
        # create alias for attributes
        for k, v  in config.items():
            setattr(self, k, v)
        if (self.n_entity is not None and self.n_relation is not None and
            self.entity_factors is not None and self.relation_factors is not None and
            self.hidden_units is not None):
            self.build()
            self.ready = True
        else:
            self.ready = False

    def fit(self, subjects, objects, predicates, batch_size=128,
            nb_epoch=10, logdir="log"):
        if not self.ready:
            if (self.n_entity is not None and self.n_relation is not None and
                self.entity_factors is not None and self.relation_factors is not None and
                self.hidden_units is not None):
                self.build()
                self.ready = True
            else:
                raise AttributeError("You must specify the architecture of the model")

        subjects = np.array(subjects, dtype=np.int64)
        objects = np.array(objects, dtype=np.int64)
        predicates = np.array(predicates, dtype=np.int64)
        if not (subjects.size == objects.size == predicates.size):
            raise ValueError("The shape of arguments to fit() is wrong")
        # add negative sample with uniform distribution
        samples = self.add_negative_samples(subjects, objects, predicates)
        subjects, objects, predicates, labels = samples
        writer = tf.train.SummaryWriter(logdir)
        cnt = 0
        for epoch in range(nb_epoch):
            indices = np.arange(subjects.size)
            np.random.shuffle(indices)
            num_loop = math.floor(subjects.size / batch_size)
            print("-----------------------------------")
            print("The {}th epoch".format(epoch+1))
            print("-----------------------------------")
            for n in range(num_loop):
                start = batch_size * n
                stop = batch_size * (n + 1)
                batch_indices = indices[start:stop]
                feed_dict = {self._subjects: subjects[batch_indices],
                            self._objects: objects[batch_indices],
                            self._predicates: predicates[batch_indices],
                            self._exist_labels: labels[batch_indices]}
                _, cost, sm = self.sess.run([self.opt, self.cost, self.summaries],
                                            feed_dict=feed_dict)
                if cnt%100 == 0:
                    writer.add_summary(sm, cnt)
                    print("The {}th loop: cost = {}".format(cnt+1, cost))
                cnt += 1
        return cost

    def add_negative_samples(self, subjects, objects, predicates):
        subjects_neg = []
        objects_neg = []
        predicate_neg = []
        nn = self.n_negative_sample
        for s, o, p in zip(subjects, objects, predicates):
            subjects_neg += [s] * nn
            objects_neg += [o] * nn
            cnt = 0
            while cnt < nn:
                p_neg = np.random.randint(0, self.n_relation)
                if p != p_neg:
                    predicate_neg.append(p_neg)
                    cnt += 1
        num_pos = len(subjects)
        num_neg = len(subjects_neg)
        labels = np.array([1] * num_pos + [-1] * num_neg)
        subjects_all = np.append(subjects, subjects_neg)
        objects_all = np.append(objects, objects_neg)
        predicate_all = np.append(predicates, predicate_neg)
        return subjects_all, objects_all, predicate_all, labels

    def _get_entity_embed_variable(self):
        entity_scale = 1.0 / np.sqrt(self.entity_factors)
        entity_embed = tf.get_variable(name="entity_embed",
                                       shape=[self.n_entity, self.entity_factors],
                                       dtype=tf.float32,
                                       initializer=tf.random_uniform_initializer(
                                       minval= -entity_scale,
                                       maxval= entity_scale))
        tf.histogram_summary("entity_embed", self._entity_embed)
        return entity_embed

    def _get_relation_embed_variable(self):
        relation_scale = 1.0 / np.sqrt(self.relation_factors) 
        relation_embed = tf.get_variable(name="relation_embed",
                                         shape=[self.n_relation, self.relation_factors],
                                         dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(
                                         minval= -relation_scale,
                                         maxval = relation_scale))
        tf.histogram_summary("relation_embed", self._relation_embed)
        return relation_embed

    def _get_relational_latent_factor(self, subject_embed, object_embed):
        """
        estimate relation embeddings between subjects and objects
        :param concat_embed: concatenated embeddings
        :return: relation embeddings
        """
        # concatenate subject and object embeddings
        concat_embed = tf.concat(1, [subject_embed, object_embed])
        activation_func = getattr(tf.nn, self.activation_func)
        self._weights = []
        self._biases = []
        # the first dim if double of entity factors, the last one is relation_factors
        units = [2 * self.entity_factors] + self.hidden_units + [self.relation_factors]
        output = concat_embed
        for layer in range(len(units)-1):
            # separate name scope for each layer to avoid name confilct
            with tf.variable_scope("hidden"+str(layer)):
                w_scale = 1.0 / np.sqrt(units[layer] * units[layer+1])
                w = tf.get_variable(name="weight",
                                    shape=[units[layer], units[layer+1]],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=w_scale))
                b = tf.Variable(np.zeros(units[layer+1]), name="bias", dtype=tf.float32)
                tf.histogram_summary("weight"+str(layer), w)
                tf.histogram_summary("bias"+str(layer), b)
                output = activation_func(tf.matmul(output, w) + b)
                self._weights.append(w)
                self._biases.append(b)
        return output


    def build(self):
        """
        build computation graph
        """
        self.graph = tf.Graph()
        with self.graph.as_default():

            # trainable variables
            self._entity_embed = self._get_entity_embed_variable()
            self._relation_embed = self._get_relation_embed_variable()


            # input nodes
            self._subjects = tf.placeholder(tf.int64, shape=[None], name="subjects")
            self._objects = tf.placeholder(tf.int64, shape=[None], name="objects")
            self._predicates = tf.placeholder(tf.int64, shape=[None], name="predicates")
            self._exist_labels = tf.placeholder(tf.float32, shape=[None], name="exist_labels")

            # look up embedding vectors
            subject_embed = tf.nn.embedding_lookup(self._entity_embed, self._subjects)
            object_embed = tf.nn.embedding_lookup(self._entity_embed, self._objects)
            predicate_embed = tf.nn.embedding_lookup(self._relation_embed, self._predicates)

            # estimated relation embeddings
            relational_embed = self._get_relational_latent_factor(subject_embed, object_embed)

            # l2 loss between relation embeddings
            l2_loss = tf.reduce_sum(tf.pow(predicate_embed - relational_embed, 2),
                                    reduction_indices=1)
            tf.histogram_summary("l2_loss", l2_loss)

            reg_term = tf.nn.l2_loss(subject_embed)
            reg_term += tf.nn.l2_loss(object_embed)
            reg_term += tf.nn.l2_loss(predicate_embed)
            for w in self._weights:
                reg_term += tf.nn.l2_loss(w)
            for b in self._biases:
                reg_term += tf.nn.l2_loss(b)


            #self.proba = proba = tf.exp(-self.sigma * l2_loss)
            #loss = tf.reduce_sum(self.sigma * l2_loss - \
            #    (1/2) * (1 - self._exist_labels) * tf.log(tf.exp(self.sigma * l2_loss) - 1 + 10e-6))

            loss = tf.reduce_sum(self._exist_labels * l2_loss)

            self.cost = (loss + self.w_regularization * reg_term)

            tf.scalar_summary("loss", loss)
            tf.scalar_summary("regularizer", reg_term)
            tf.scalar_summary("regularizer_scaled", reg_term*self.w_regularization)
            tf.scalar_summary("cost", self.cost)

            self.summaries = tf.merge_all_summaries()

            optimizer = getattr(tf.train, OPTIMIZER_NAME[self.optimizer])
            self.opt = optimizer(self.learning_rate).minimize(self.cost)

            self.sess = tf.Session()
            self.init_op = tf.initialize_all_variables()
            self.sess.run(self.init_op)


    def build_assign_ops(self):
        # Assign operation
        self.input_entity_embed = tf.placeholder("float32", shape=self._entity_embed.get_shape())
        self.assign_entity_embed = self._entity_embed.assign(self.input_entity_embed)
        self.input_relation_embed = tf.placeholder("float32", shape=self._relation_embed.get_shape())
        self.assign_relation_embed = self._relation_embed.assign(self.input_relation_embed)

        self.input_weight = []
        self.input_bias = []
        self.assign_weight = []
        self.assign_bias = []

        for w in self._weights:
            input_w = tf.placeholder("float32", shape=w.get_shape())
            self.input_weight.append(input_w)
            self.assign_weight.append(w.assign(input_w))

        for b in self._biases:
            input_b = tf.placeholder("float32", shape=b.get_shape())
            self.input_bias.append(input_b)
            self.assign_bias.append(b.assign(input_b))


    def get_entity_embedding(self, indices=None):
        entity_embed = self.sess.run(self._entity_embed)
        if indices:
            return entity_embed[indices, :]
        else:
            return entity_embed

    def get_relation_embedding(self, indices=None):
        relation_embed = self.sess.run(self._relation_embed)
        if indices:
            return relation_embed[indices, :]
        else:
            return relation_embed

    def set_entity_embedding(self, entity_embed):
        self.sess.run(self.assign_entity_embed,
                      feed_dict={self.input_entity_embed: entity_embed})

    def set_relation_embedding(self, relation_embed):
        self.sess.run(self.assign_relation_embed,
                      feed_dict={self.input_relation_embed: relation_embed})

    def get_estimated_embedding(self, subjects, objects):
        subjects = np.array(subjects, dtype=np.int64)
        objects = np.array(objects, dtype=np.int64)
        feed_dict = {self._subjects: subjects, self._objects: objects}
        res = self.sess.run(self._estimated_embed, feed_dict=feed_dict)
        return res

    def save(self, save_file):
        stored = {}
        entity_embed = self.get_entity_embedding()
        relation_embed = self.get_relation_embedding()
        weights = self.get_weights()
        biases = self.get_biases()
        stored["entity_embed"] = entity_embed
        stored["relation_embed"] = relation_embed
        stored["weights"] = weights
        stored["biases"] = biases
        stored["config"] = self.config
        joblib.dump(stored, save_file)

    @classmethod
    def restore(cls, save_file):
        stored = joblib.load(save_file)
        config = stored["config"]
        model = cls(**config)
        model.set_entity_embedding(stored["entity_embed"])
        model.set_relation_embedding(stored["relation_embed"])
        model.set_weights(stored["weights"])
        model.set_biases(stored["biases"])
        return model


    def get_weights(self):
        return [self.sess.run(w) for w in self._weights]

    def get_biases(self):
        return [self.sess.run(b) for b in self._biases]


    def set_weights(self, weights):
        for i, weight_i in enumerate(weights):
            if weight_i.shape != tuple(self._weights[i].get_shape().as_list()):
                raise ValueError("The shape of weight in layer {}th is invalid".format(i))
            self.sess.run(self.assign_weight[i], {self.input_weight[i]: weight_i})

    def set_biases(self, biases):
        for i, bias_i in enumerate(biases):
            if bias_i.shape != tuple(self._biases[i].get_shape().as_list()):
                raise ValueError("The shape of bias in layer {}th is invalid".format(i))
            self.sess.run(self.assign_bias[i], {self.input_bias[i]: bias_i})
