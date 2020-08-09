import tensorflow as tf
import numpy as np
from settings import *
from replay_memory import RLMemory
from network import BestRespQDN
from random import uniform


class Agent:

    def __init__(self, env, i, act_dim, critic_lr, batch_size, nodes, dr, ep):
        self.env = env
        self.id = i
        self.count = 0
        self.batch_size = batch_size
        self.reward = 0.
        self.rev = 0.
        self.dr = dr
        self.ep = ep
        self.a_dim = act_dim
        self.action = -1
        self.joint_action = [0. for _ in range(act_dim)]
        self.sess = tf.Session()
        self.nw = BestRespQDN(self.sess, act_dim, critic_lr, nodes)
        self.memory = RLMemory()
        self.sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()

    def act(self, ja):
        rand = uniform(0, 1)
        self.joint_action = [v for v in ja]
        q_op = self.nw.predict([ja], self.dr)[0]
        if rand > self.ep:
            # argmax action
            # rand = uniform(0, 1)
            # a = np.argmax(q_op) if rand > self.ep else randint(0, self.a_dim - 1)

            # softmax action
            q_op = q_op - np.max(q_op)
            q_op = np.exp(q_op) / np.sum(np.exp(q_op))
            a = np.random.choice([j for j in range(np.size(q_op))], 1, p=q_op)[0]
        else:
            # sample from suggested joint action
            a = np.random.choice([j for j in range(np.size(ja))], 1, p=ja)[0]
        self.action = a
        return a

    def update_reward(self):
        self.reward = self.env.get_reward(self.action)
        self.rev += self.reward

    def update_learning(self):
        r = self.reward
        a_ip = np.zeros(self.a_dim)
        a_ip[self.action] = 1
        # if self.count % 2000 > 1990:
        #     state_v = self.nw.predict([self.joint_action], self.kp)[0]
        #     q_loss = self.nw.get_loss([a_ip], [self.joint_action], [[r]], self.kp)
        #     print("\n d", self.id, self.count, "action:", act, "reward:", r,
        #           "\nq loss:", q_loss, "joint_act", self.joint_action, "\n q-value:", state_v, "\r\n")

        self.memory.add(a_ip, [r/10.], [v for v in self.joint_action])

        self.reward = 0.0
        if self.ep == MIN_EP:
            return

        # train the network
        if self.count % TRAINING_STEPS == 0 and self.memory.size() >= self.batch_size:
            action_batch, r_batch, ja_batch = self.memory.sample(self.batch_size)
            self.nw.train(ja_batch, action_batch, r_batch, self.dr)

    def update(self, f):
        self.count += 1
        if self.count % EP_COUNT == 0:
            self.ep *= DECAY if self.ep > MIN_EP else MIN_EP
        if self.count % COUNT == 0:
            # print(self.count / COUNT, self.id, self.rev)
            f.write(str(self.count / COUNT) + "," + str(self.id) + "," + str(self.rev) + "\n")
            f.flush()
            self.rev = 0

        if self.count % 1.2e7 == 0:
            # set dropout rate to zero
            self.dr = 0
        if self.count > 20000000:
            exit(0)
