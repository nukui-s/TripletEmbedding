# -*- utf-8 -*-

import math

import numpy as np
import tensorflow as tf
import logging


OPTIMIZER_NAME = {"adam": "AdamOptimizer",
                  "sgd": "GradientDescentOptimizer"}


class TripletEmbedding(object):
    """
    Tripet Embedding model
    """

    def __init__(self, n_entity, n_relation, entity_factors, relation_factors,
                 hidden_units, lambda_w=10e-3, sigma=1.0, activation_func="relu",
                 optimizer="adam", learning_rate=10e-2, w_regularization=10e-4):
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
        """
        if not hasattr(tf.nn, activation_func):
            raise("TensorFlow has no such activation func"
                  .format(activation_func))
        if not hasattr(tf.train, OPTIMIZER_NAME[optimizer]):
            raise("TensorFlow has no such optimizer {}".format(optimizer))

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.hidden_units = hidden_units
        self.w_regularization = w_regularization
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.entity_factors = entity_factors
        self.relation_factors = relation_factors
        self.activation_func = activation_func
        self.optimizer = OPTIMIZER_NAME[optimizer]
        self.build()

    def fit(self, subjects, objects, predicates, labels, batch_size=128):
        subjects = np.array(subjects, dtype=np.int64)
        objects = np.array(objects, dtype=np.int64)
        predicates = np.array(predicates, dtype=np.int64)
        labels = np.array(labels, dtype=np.float32)
        if not (subjects.size == objects.size ==
                predicates.size == labels.size):
            raise ValueError("The shape of arguments to fit() is wrong")
        num_loop = math.floor(subjects.size / batch_size)
        for n in range(num_loop):
            start = batch_size * n
            stop = batch_size * (n + 1)
            feed_dict = {self._subjects: subjects[start:stop],
                         self._objects: objects[start:stop],
                         self._predicates: predicates[start:stop],
                         self._exist_labels: labels[start:stop]}
            _, cost = self.sess.run([self.opt, self.cost], feed_dict=feed_dict)
            logging.info("The {}th loop: cost = {}".format(n, cost))
        return cost


    def build(self):
        """
        build computation graph
        """
        self.graph = tf.Graph()
        with self.graph.as_default():

            # trainable variables
            self._entity_embed = tf.get_variable(name="entity_embed",
                                                 shape=[self.n_entity, self.entity_factors],
                                                 dtype=tf.float32,
                                                 initializer=tf.random_uniform_initializer())
            self._relation_embed = tf.get_variable(name="relation_embed",
                                                  shape=[self.n_relation, self.relation_factors],
                                                  dtype=tf.float32,
                                                  initializer=tf.random_uniform_initializer())
            # input nodes
            self._subjects = tf.placeholder(tf.int64, shape=[None], name="subjects")
            self._objects = tf.placeholder(tf.int64, shape=[None], name="objects")
            self._predicates = tf.placeholder(tf.int64, shape=[None], name="predicates")
            self._exist_labels = tf.placeholder(tf.float32, shape=[None], name="exist_labels")

            # look up embedding vectors
            subject_embed = tf.nn.embedding_lookup(self._entity_embed, self._subjects)
            object_embed = tf.nn.embedding_lookup(self._entity_embed, self._objects)
            predicate_embed = tf.nn.embedding_lookup(self._relation_embed, self._predicates)

            # concatenate subject and object embeddings
            concat_embed = tf.concat(1, [subject_embed, object_embed])

            # estimated relation embeddings
            self._estimated_embed = self.estimate_relation_embeddings(concat_embed)

            # l2 loss between relation embeddings
            l2_loss = tf.reduce_sum(tf.pow(predicate_embed - self._estimated_embed, 2),
                                    reduction_indices=1)

            self.proba = proba = tf.exp(-self.sigma * l2_loss)
            self.cost = cost = tf.reduce_sum(self.sigma * l2_loss - \
                    (1/2) * (1 - self._exist_labels) * tf.log(tf.exp(self.sigma * l2_loss) - 1 + 10e-6))

            optimizer = getattr(tf.train, self.optimizer)
            self.opt = optimizer(self.learning_rate).minimize(self.cost)

            self.sess = tf.Session()
            self.init_op = tf.initialize_all_variables()
            self.sess.run(self.init_op)


    def estimate_relation_embeddings(self, concat_embed):
        """
        estimate relation embeddings between subjects and objects
        :param concat_embed: concatenated embeddings
        :return: relation embeddings
        """
        activation_func = getattr(tf.nn, self.activation_func)
        self._weights = []
        self._biases = []
        # the first dim if double of entity factors, the last one is relation_factors
        units = [2 * self.entity_factors] + self.hidden_units + [self.relation_factors]
        output = concat_embed
        for layer in range(len(units)-1):
            # separate name scope for each layer to avoid name confilct
            with tf.variable_scope("hidden"+str(layer)):
                w = tf.get_variable(name="weight",
                                    shape=[units[layer], units[layer+1]],
                                    dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer())
                b = tf.Variable(np.zeros(units[layer+1]), name="bias", dtype=tf.float32)
                output = activation_func(tf.matmul(output, w) + b)
                self._weights.append(w)
                self._biases.append(b)
        return output


