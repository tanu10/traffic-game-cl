from settings import *
from agent import *
from resource import *
from replay_memory import CentralAgentMemory
from network import DDPGActorNetwork, DDPGCriticNetwork


class Environment:
    def __init__(self):
        self.count = 1
        self.num_agent = NUM_AGENT
        self.num_resources = NUM_RESOURCE
        self.batch_size = BATCH_SIZE
        self.kp = DROPOUT_RATE
        self.ep = EPSILON
        self.reward = 0.
        self.resource_reward = [0. for _ in range(self.num_resources)]
        self.rev = 0.
        self.agents = []
        self.resources = []
        self.current_policy = []
        self.actual_act = [0. for _ in range(self.num_resources)]
        self.memory = CentralAgentMemory()
        self.sess = tf.Session()
        self.actor = DDPGActorNetwork(self.sess, 1, self.num_resources, ACTOR_LR, TAU, BATCH_SIZE, INTERMEDIATE_NODES)
        self.critic = DDPGCriticNetwork(self.sess, 1, self.num_resources,
                                        CRITIC_LR, TAU, self.actor.get_num_trainable_vars(), INTERMEDIATE_NODES)
        self.sess.run(tf.global_variables_initializer())
        self.init_agents()

    def init_agents(self):
        for i in range(NUM_AGENT):
            self.agents.append(Agent(self, i, self.num_resources, CRITIC_LR, BATCH_SIZE, INTERMEDIATE_NODES, DROPOUT_RATE,
                                     EPSILON))
        for i in range(self.num_resources):
            self.resources.append(Resource(i, MU[i], SIGMA[i]))

    def run(self):
        while True:
            if self.count > MAX_COUNT:
                exit()
            self.act()
            self.update_reward()
            self.update_learning()
            self.update()

    def update(self):
        self.count += 1
        if self.count % EP_COUNT == 0:
            self.ep = max(self.ep*DECAY, MIN_EP)
        if self.count % COUNT == 0:
            # print(self.count / COUNT, self.rev)
            central_f.write(str(self.count / COUNT) + "," + str(self.rev) + "\n")
            central_f.flush()
            self.rev = 0.
        for agent in self.agents:
            agent.update(agent_f)

    def update_learning(self):
        if self.ep > MIN_EP:
            act = [v for v in self.actual_act]
            self.memory.add(act, [self.reward/10.])
            self.reward = 0.

            if self.count % TRAINING_STEPS == 0:
                s_batch = [[1]] * self.batch_size
                a_batch, r_batch = self.memory.sample(self.batch_size)
                l, _ = self.critic.train(s_batch, a_batch, r_batch, self.kp)
                # if self.count % 20000 > 19950:
                #     print("Central loss", l)
                a_outs = self.actor.predict(s_batch, self.kp)
                grads = self.critic.action_gradients(s_batch, a_outs)
                self.actor.train(s_batch, grads[0], self.kp)
            if self.count % TARGET_STEPS:
                self.actor.update_target_network()

        # let agents update their learning
        for agent in self.agents:
            agent.update_learning()

    def act(self):
        self.current_policy = self.actor.predict([[1]], self.kp)[0]
        self.actual_act = [0. for _ in range(self.num_resources)]
        for agent in self.agents:
            a = agent.act(self.current_policy)
            self.actual_act[a] += 1
        for rid in range(self.num_resources):
            self.resource_reward[rid] = self.resources[rid].get_cost(self.actual_act[rid])
        self.reward = np.sum(self.resource_reward)
        self.rev += self.reward
        self.actual_act = [v / self.num_agent for v in self.actual_act]
        if self.count % 2000 > 1990:
            print("Cumulative reward", self.reward, "suggested act:", self.current_policy,
                  "actual act:", self.actual_act, "epsilon:", self.agents[0].ep)

    def update_reward(self):
        for agent in self.agents:
            agent.update_reward()

    def get_reward(self, rid):
        return self.resource_reward[rid]


agent_f = open(AGENT_LOG, 'w')
central_f = open(CENTRAL_AGENT_LOG, 'w')
mf = Environment()
mf.run()
