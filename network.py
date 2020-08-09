import tensorflow as tf


class BestRespQDN:

    def __init__(self, sess, act_dim, lr, nodes):
        self.sess = sess
        # self.s_dim = state_dim
        self.a_dim = act_dim
        self.learning_rate = lr
        self.nodes = nodes
        self.dr = tf.placeholder(tf.float32)
        self.predicted_q_value = tf.placeholder("float", [None, 1], name="prediction_batch")
        self.action = tf.placeholder("float", [None, self.a_dim], name="action_batch")

        self.q_out, self.input_action = self.create_network("dqn")
        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - tf.reduce_sum(self.q_out * self.action,
                                                                                    axis=1, keepdims=True)))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def create_network(self, name):
        with tf.variable_scope(name):
            suggested_ja = tf.placeholder("float", [None, self.a_dim], name="suggested_join__action_batch")
            h0 = tf.layers.dense(inputs=suggested_ja, units=self.nodes, activation=tf.nn.relu)
            h0 = tf.contrib.layers.layer_norm(h0)
            h0 = tf.layers.dropout(h0, self.dr)
            h1 = tf.layers.dense(inputs=h0, units=self.nodes, activation=tf.nn.relu)
            h1 = tf.contrib.layers.layer_norm(h1)
            h1 = tf.layers.dropout(h1, self.dr)
            out = tf.layers.dense(inputs=h1, units=self.a_dim, activation=None)
            return out, suggested_ja

    def train(self, input_action, action, predicted_q_value, dr):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.input_action: input_action,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.dr: dr
        })

    def predict(self, joint_act, dr):
        return self.sess.run(self.q_out, feed_dict={
            self.input_action: joint_act,
            self.dr: dr
        })

    def get_loss(self, act, joint_act, val, dr):
        return self.sess.run(self.loss, feed_dict={
            self.action: act,
            self.input_action: joint_act,
            self.predicted_q_value: val,
            self.dr: dr
        })

    # def predict_target(self, state, zone, joint_act, dr):
    #     return self.sess.run(self.target_q_out, feed_dict={
    #         self.next_state: state,
    #         self.next_zone: zone,
    #         self.next_input_action: joint_act,
    #         self.keep_prob: dr
    #     })
    #
    # def update_target_network(self):
    #     self.sess.run(self.update_op)


class DDPGActorNetwork(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size, nodes):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.nodes = nodes
        self.batch_size = batch_size

        self.state = tf.placeholder("float", [None, self.s_dim], name="state_batch")
        self.target_state = tf.placeholder("float", [None, self.s_dim], name="state_batch")
        self.dr = tf.placeholder(tf.float32)

        # Actor Network
        self.action = self.create_actor_network(self.state)

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_action = self.create_actor_network(self.target_state)

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(self.action, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients,
                                                                                       self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self, inputs):

        h0 = tf.layers.dense(inputs=inputs, units=self.nodes, activation=tf.nn.relu)
        h0 = tf.contrib.layers.layer_norm(h0)
        h0 = tf.layers.dropout(h0, self.dr)
        h1 = tf.layers.dense(inputs=h0, units=self.nodes, activation=tf.nn.relu)
        h1 = tf.contrib.layers.layer_norm(h1)
        h1 = tf.layers.dropout(h1, self.dr)
        w_init = tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3)
        h = tf.layers.dense(inputs=h1, units=self.a_dim, kernel_initializer=w_init, activation=tf.nn.softmax)
        return h

    def train(self, state, a_gradient, dr):
        self.sess.run(self.optimize, feed_dict={
            self.state: state,
            self.action_gradient: a_gradient,
            self.dr: dr
        })

    def predict(self, state, dr):
        return self.sess.run(self.action, feed_dict={
            self.state: state,
            self.dr: dr
        })

    def predict_target(self, state, dr):
        return self.sess.run(self.target_action, feed_dict={
            self.target_state: state,
            self.dr: dr
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class DDPGCriticNetwork(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars, nodes):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.nodes = nodes

        self.dr = tf.placeholder(tf.float32)
        # Create the critic network
        self.state, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1], name="critic_predicted_qval")

        # Define loss and optimization op
        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.out))
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        state = tf.placeholder("float", [None, self.s_dim], name="critic_state_batch")
        action = tf.placeholder("float", [None, self.a_dim], name="critic_action_batch")

        h0 = tf.layers.dense(inputs=state, units=self.nodes, activation=tf.nn.relu)
        h0 = tf.contrib.layers.layer_norm(h0)
        h0 = tf.layers.dropout(h0, self.dr)
        # Add the action in the 2nd layer
        h1 = tf.layers.dense(inputs=action, units=self.nodes, activation=tf.nn.relu)
        h1 = tf.concat(values=[h1, h0], axis=1)
        h1 = tf.contrib.layers.layer_norm(h1)
        h1 = tf.layers.dropout(h1, self.dr)

        w_init = tf.initializers.random_uniform(minval=-3e-3, maxval=3e-3)
        out = tf.layers.dense(inputs=h1, units=1, kernel_initializer=w_init, activation=None)
        return state, action, out

    def train(self, state, action, predicted_q_value, dr):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.state: state,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.dr: dr
        })

    def predict(self, inputs, action, dr):
        return self.sess.run(self.out, feed_dict={
            self.state: inputs,
            self.action: action,
            self.dr: dr
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: inputs,
            self.action: actions
        })

