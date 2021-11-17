import threading

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from time import sleep
from copy import deepcopy
import os
import time
import argparse

import LS_env as env
from scipy.stats import poisson




# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer





class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer,depth_nn_out,activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor):
        with tf.variable_scope(scope):
            self.entropy_factor = entropy_factor
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)


            if depth_nn_hidden >= 1:
                self.hidden1 = slim.fully_connected(inputs=self.inputs, num_outputs=depth_nn_layers_hidden[0],
                                                    activation_fn=activation_nn_hidden[0])

                self.state_out = slim.fully_connected(inputs=self.hidden1, num_outputs=depth_nn_out,
                                                      activation_fn=activation_nn_out)
            if depth_nn_hidden >= 2:
                self.hidden2 = slim.fully_connected(inputs=self.hidden1, num_outputs=depth_nn_layers_hidden[1],
                                                    activation_fn=activation_nn_hidden[1])
                self.state_out = slim.fully_connected(inputs=self.hidden2, num_outputs=depth_nn_out,
                                                      activation_fn=activation_nn_out)

            if depth_nn_hidden >= 3:
                self.hidden3 = slim.fully_connected(inputs=self.hidden2, num_outputs=depth_nn_layers_hidden[2],
                                                    activation_fn=activation_nn_hidden[2])
                self.state_out = slim.fully_connected(inputs=self.hidden3, num_outputs=depth_nn_out,
                                                      activation_fn=activation_nn_out)
            if depth_nn_hidden >= 4:
                self.hidden4 = slim.fully_connected(inputs=self.hidden3, num_outputs=depth_nn_layers_hidden[3],
                                                    activation_fn=activation_nn_hidden[3])
                self.state_out = slim.fully_connected(inputs=self.hidden4, num_outputs=depth_nn_out,
                                                      activation_fn=activation_nn_out)

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(self.state_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)


            self.value = slim.fully_connected(self.state_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy =  -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10))
                self.policy_loss = tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.25 * self.value_loss + self.policy_loss - self.entropy * self.entropy_factor

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Worker():
    def __init__(self, name, s_size, a_size, trainer, model_path, best_path,log_path, global_episodes,depth_nn_out,activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.best_path = best_path
        self.log_path = log_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.no_improvement = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(
            model_path + str(self.number) + str(time.strftime(" %Y%m%d-%H%M%S")))
        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer,depth_nn_out,activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor)
        self.update_local_ops = update_target_graph('global', self.name)
        self.actions =  np.identity(a_size, dtype=bool).tolist()
        self.bool_evaluating = None
        self.best_mean_solution = np.inf
        self.mean_solution_vector = []
        self.median_solution_vector = []

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages  # ,
                     }

        v_l, p_l, e_l, g_n, v_n, Policy, _ = sess.run(
            [self.local_AC.value_loss,
             self.local_AC.policy_loss,
             self.local_AC.entropy,
             self.local_AC.grad_norms,
             self.local_AC.var_norms,
             self.local_AC.policy,
             self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, saver_best, LT_s, b, C_f, InvMax, initial_state,
             max_training_episodes,actions,p_len_episode_buffer,max_no_improvement,warmup,
             best_path,LT_min, LT_max,eval_length):
        episode_count = sess.run(self.global_episodes)
        self.no_improvement = 0
        nb_parallel = args.parallel_evals
#


        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            start = time.time()
            mean_performance = np.inf
            self.best_mean_solution = np.inf
            while episode_count < max_training_episodes:
                if (episode_count % args.evalfrequency == 0 and episode_count > 10 and mean_performance < 1.05*self.best_mean_solution):
                    self.bool_evaluating = True

                elif (episode_count % args.evalfrequency == 0 and episode_count > 10 and mean_performance > 1.05*self.best_mean_solution):
                    self.no_improvement+=1
                    self.bool_evaluating = None

                else:
                    self.bool_evaluating = None

                eval_performance = []


                for i in range(nb_parallel):
                    episode_buffer = []
                    episode_values = []

                    episode_reward = 0
                    episode_step_count = 0
                    d = False

                    self.inv_vect = initial_state

                    s = deepcopy(self.inv_vect)
                    q_arrivals = np.ones(len(self.inv_vect))

                    if self.bool_evaluating:
                        nb_episodes = eval_length

                    else:
                        nb_episodes = max_episode_length

                    while (episode_step_count < nb_episodes - 1):
                            #Rescale state to fall between 0 and 1:
                            s2 = s/(InvMax+LT_s*args.max_order)
                            s2 = np.append(s2,1)

                            # Take an action using probabilities from policy network output.
                            a_dist, v = sess.run([self.local_AC.policy, self.local_AC.value],
                                                 feed_dict={self.local_AC.inputs: [s2.flatten()]})  # ,

                            a = np.random.choice(np.arange(len(a_dist[0])), p=a_dist[0])



                            if self.bool_evaluating == True:
                                demand = np.random.choice(np.arange(len(demand_probabilities)), p=demand_probabilities)
                                #demand = [sample_paths[i][episode_step_count]]
                                #r, s1 = env.transition(s, actions[a], demand, args)
                                #LT = LTs[i][episode_step_count]
                                LT = np.random.randint(LT_min, LT_max+1)

                                r, s1,q_arrivals = env.transition_stochLT(s, actions[a], demand, q_arrivals, args,LT)

                            else:
                                demand = np.random.choice(np.arange(len(demand_probabilities)), p=demand_probabilities)
                                #r, s1 = env.transition(s, actions[a], demand, args)
                                LT = np.random.randint(LT_min, LT_max+1)
                                r, s1, q_arrivals = env.transition_stochLT(s, actions[a], demand, q_arrivals, args, LT)
                                d = False
                            episode_buffer.append([s2.flatten(), a, r, s1, d, v[0, 0]])

                            episode_values.append(v[0, 0])
                            if(episode_step_count> warmup*nb_episodes):
                                episode_reward += r
                            s = deepcopy(s1)
                            episode_step_count += 1

                            # If the episode hasn't ended, but the experience buffer is full, then we
                            # make an update step using that experience rollout.
                            if len(episode_buffer) == p_len_episode_buffer and episode_step_count != nb_episodes - 1 and self.bool_evaluating != True:

                                # Since we don't know what the true final return is, we "bootstrap" from our current
                                # value estimation.
                                s2 = s/(InvMax+LT_s*args.max_order)
                                s2 = np.append(s2,1)
                                v1 = sess.run(self.local_AC.value,
                                              feed_dict={self.local_AC.inputs: [s2.flatten()]})[0, 0]
                                v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)

                                episode_buffer = []
                                sess.run(self.update_local_ops)

                    eval_performance.append(episode_reward/(episode_step_count-warmup*nb_episodes))
                if(self.bool_evaluating):
                    print(self.number,np.mean(eval_performance),np.std(eval_performance))
                mean_performance = np.mean(eval_performance)

                print(time.time() - start)

                if (mean_performance < self.best_mean_solution and self.bool_evaluating == True):
                    self.best_mean_solution = mean_performance
                    self.mean_solution_vector = eval_performance
                    self.no_improvement = 0
                    print('changed',self.no_improvement,self.number)

                    with open(best_path+'best_mean_solution%i-%i.csv'%(args.setting,self.number), 'a') as f:
                        f.write(str(self.best_mean_solution) + ';'+ str(np.std(self.mean_solution_vector)) + ';' + str(LT_s) + ';'+ str(b) + ';'+ str(C_f) + ';')
                        for item in self.mean_solution_vector:
                            f.write(str(item) + ';')
                        f.write('\n')
                    saver_best.save(sess, self.best_path + '/Train_' + str(
                        self.number) + '/model_mean_setting' + '-' +str(args.setting)+ '-'+str(self.number)+ '.cptk')


                elif self.bool_evaluating == True:
                    self.no_improvement += 1

                if self.name == 'worker_0':
                    sess.run(self.increment)

                print('no improvement for: ',self.no_improvement,' periods','best:',self.best_mean_solution,'current: ',mean_performance ,self.number)

                episode_count += 1
                if self.no_improvement >= max_no_improvement:
                    break


def write_parameters(model_path, depth_nn_hidden, depth_nn_layers_hidden, depth_nn_out, entropy_factor,
                     activation_nn_hidden, activation_nn_out, learning_rate, optimizer, activations,
                     p_len_episode_buffer, max_episode_length, OrderFast, OrderSlow, LT_s, LT_f, InvMax,
                     max_training_episodes,h,b,C_f,C_s,InvMin,initial_state,nb_workers,warmup):
    f = open(model_path + "/Parameters.txt", "w")
    parameters = {}
    f.write("depth_nn_hidden: " + str(depth_nn_hidden))
    parameters["depth_nn_hidden"] = depth_nn_hidden
    f.write("\ndepth_nn_layers_hidden " + str(depth_nn_layers_hidden))
    parameters["depth_nn_layers_hidden"] = depth_nn_layers_hidden
    f.write("\ndepth_nn_out: " + str(depth_nn_out))
    parameters["depth_nn_out"] = depth_nn_out
    f.write("\nentropy_factor " + str(entropy_factor))
    parameters["entropy_factor"] = entropy_factor
    f.write("\nactivation_nn_hidden: " + str(activation_nn_hidden))
    parameters["activation_nn_hidden"] = activation_nn_hidden
    f.write("\nactivation_nn_out " + str(activation_nn_out))
    parameters["activation_nn_out"] = activation_nn_out
    f.write("\nLearning Rate: " + str(learning_rate))
    parameters["learning_rate"] = learning_rate
    f.write("\noptimizer " + str(optimizer))
    parameters["optimizer"] = optimizer
    f.write("\nactivations: " + str(activations))
    parameters["activations"] = activations
    f.write("\np_len_episode_buffer " + str(p_len_episode_buffer))
    parameters["p_len_episode_buffer"] = p_len_episode_buffer
    f.write("\nmax_episode_length: " + str(max_episode_length))
    parameters["max_episode_length"] = max_episode_length
    f.write("\nOrderFast " + str(OrderFast))
    parameters["OrderFast"] = OrderFast
    f.write("\nOrderSlow " + str(OrderSlow))
    parameters["OrderSlow"] = OrderSlow
    f.write("\nLT_s " + str(LT_s))
    parameters["LT_s"] = LT_s
    f.write("\nLT_f " + str(LT_f))
    parameters["LT_f"] = LT_f
    f.write("\nh " + str(h))
    parameters["h"] = h
    f.write("\nb " + str(b))
    parameters["b"] = b
    f.write("\nC_f " + str(C_f))
    parameters["C_f"] = C_f
    f.write("\nC_s " + str(C_s))
    parameters["C_s"] = C_s
    f.write("\nInvMin " + str(InvMin))
    parameters["InvMin"] = InvMin
    f.write("\nInvMax " + str(InvMax))
    parameters["InvMax"] = InvMax
    f.write("\ninitial_state " + str(initial_state))
    parameters["initial_state"] = initial_state
    f.write("\nmax_training_episodes " + str(max_training_episodes))
    parameters["max_training_episodes"] = max_training_episodes
    f.write("\nnb_workers" + str(nb_workers))
    parameters["nb_workers"] = nb_workers
    f.write("\nwarmup" + str(warmup))
    parameters['warmup'] = warmup
    f.close()
    return parameters



def objective(parameters):
    Demand_Max = parameters['Demand_Max']
    OrderFast = parameters['OrderFast']
    OrderSlow = parameters['OrderSlow']
    LT_f = parameters['LT_f']
    LT_s = parameters['LT_s']
    h = parameters['h']
    b = parameters['b']
    C_f = parameters['C_f']
    C_s = parameters['C_s']
    max_training_episodes = parameters['max_training_episodes']
    learning_rate = parameters['initial_lr']
    entropy_factor = parameters['entropy']
    gamma = parameters['gamma']
    max_no_improvement = parameters['max_no_improvement']
    max_training_episodes = parameters['max_training_episodes']
    depth_nn_hidden = parameters['depth_nn_hidden']
    depth_nn_layers_hidden = parameters['depth_nn_layers_hidden']
    depth_nn_out = parameters['depth_nn_out']
    p_len_episode_buffer = parameters['p_len_episode_buffer']
    InvMax = parameters['inv_max']
    InvMin = parameters['invmin']
    training = parameters['training']
    nb_workers = parameters['nbworkers']
    warmup = parameters['warmup']
    activation_nn_hidden = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]
    activation_nn_out = tf.nn.relu
    optimizer = tf.train.AdamOptimizer(learning_rate)
    max_episode_length = parameters['max_episode_length']

    multiplier = parameters['initial_state']
    initial_state = np.ones(LT_s)*multiplier
    LT_min = parameters['LT_min']
    LT_max = parameters['LT_max']
    eval_length = parameters['eval_length']

    actions = env.actions(args)
    a_size = len(actions)
    s_size = len(initial_state) + 1

    tf.reset_default_graph()
    if training:
        load_model = False
    else:
        load_model = True


    model_path = 'Logs/Setting%i/buf%d/lr%d/entr%d/Logs_'%(args.setting,args.p_len_episode_buffer,1/args.initial_lr,1/args.entropy) + str(time.strftime("%Y%m%d-%H%M%S")) + '/model'
    best_path = 'Logs/Setting%i/buf%d/lr%d/entr%d/Logs_'%(args.setting,args.p_len_episode_buffer,1/args.initial_lr,1/args.entropy) + str(time.strftime("%Y%m%d-%H%M%S")) + '/best'
    log_path = 'Logs/'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = optimizer
    master_network = AC_Network(s_size, a_size, 'global', None,depth_nn_out,activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor)  # Generate global network
    num_workers = nb_workers
    workers = []

    for i in range(num_workers):
        if not os.path.exists(best_path + '/Train_' + str(i)):
            os.makedirs(best_path + '/Train_' + str(i))
        workers.append(Worker(i, s_size, a_size, trainer, model_path, best_path,log_path, global_episodes,depth_nn_out,activation_nn_hidden,depth_nn_hidden,depth_nn_layers_hidden,activation_nn_out,entropy_factor))
    saver = tf.train.Saver(max_to_keep=5)
    saver_best = tf.train.Saver(max_to_keep=None)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state('./')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        # Start the "work" process for each worker in a separate threat.
        worker_threads = []
        temp_best_mean_solutions = np.zeros(len(workers))
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length, gamma, sess, saver_best, LT_s, b, C_f, InvMax, initial_state,
            max_training_episodes, actions, p_len_episode_buffer, max_no_improvement, warmup,
            best_path, LT_min, LT_max,eval_length)


            t = threading.Thread(target=(worker_work))
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)
        for index, worker in enumerate(workers):
            temp_best_mean_solutions[index] = worker.best_mean_solution

        best_mean_solution_found = np.min(temp_best_mean_solutions)
        with open(log_path+'best_mean_solution.csv','a') as f:
            f.write(str(best_mean_solution_found)+';'+str(best_path)+';')
            for key,value in parameters.items():
                f.write(str(key)+';')
            f.write('\n')
            f.write(str(best_mean_solution_found) + ';' + str(best_path) + ';')
            for key, value in parameters.items():
                f.write(str(value) + ';')
            for item in worker.mean_solution_vector:
                f.write(str(item) + ';')
            f.write('\n')

        return best_mean_solution_found

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--initial_lr', default=0.0001, type=float,
                        help="Initial value for the learning rate.  Default = 0.0001",
                        dest="initial_lr")
    parser.add_argument('--entropy', default=0.0000001, type=float,
                        help="Strength of the entropy regularization term (needed for actor-critic). Default = 0.000001",
                        dest="entropy")
    parser.add_argument('--gamma', default=0.999, type=float, help="Discount factor. Default = 0.99", dest="gamma")
    parser.add_argument('--max_no_improvement', default=30, type=float, help="max_no_improvement. Default = 5000", dest="max_no_improvement")
    parser.add_argument('--max_training_episodes', default=1000000, type=float, help="max_training_episodes. Default = 10000000",
                        dest="max_training_episodes")
    parser.add_argument('--depth_nn_hidden', default=3, type=float,
                        help="depth_nn_hidden. Default = 3",
                        dest="depth_nn_hidden")
    parser.add_argument('--depth_nn_out', default=20, type=float,
                        help="depth_nn_out. Default = 20",
                        dest="depth_nn_out")
    parser.add_argument('--depth_nn_layers_hidden', default=[150,120,80,0], type=list,
                        help="depth_nn_layers_hidden. Default = [150,120,80,0]",
                        dest="depth_nn_layers_hidden")
    parser.add_argument('--p_len_episode_buffer', default=20, type=float,
                        help="p_len_episode_buffer. Default = 20",
                        dest="p_len_episode_buffer")
    parser.add_argument('--initial_state', default=[4], type=float,
                        help="initial_state. Default = [3,0]",
                        dest="initial_state")
    parser.add_argument('--inv_max', default=100, type=float,
                        help="inv_max. Default = 150",
                        dest="inv_max")
    parser.add_argument('--invmin', default=-10 , type=float,
                        help="invmin. Default = -15",
                        dest="invmin")


    parser.add_argument('--training', default= True, type=float,
                        help="training. Default = True",
                        dest="training")

    parser.add_argument('--nbworkers', default= 4, type=int,
                        help="Number of A3C workers. Default = 4",
                        dest="nbworkers")

    parser.add_argument('--OrderFast', default=4, type=int,
                        help="OrderFast. Default = 5",
                        dest="OrderFast")
    parser.add_argument('--OrderSlow', default=4, type=int, help="OrderSlow. Default = 5", dest="OrderSlow")

    parser.add_argument('--LT_s', default=4, type=int, help="LT_s. Default = 1", dest="LT_s")
    parser.add_argument('--LT_f', default=4, type=int, help="LT_f. Default = 0",
                        dest="LT_f")
    parser.add_argument('--LT_min', default=2, type=int, help="LT_min. Default = 2",
                        dest="LT_min")
    parser.add_argument('--LT_max', default=4, type=int, help="LT_max. Default = 4",
                        dest="LT_max")

    parser.add_argument('--C_s', default=1, type=float,
                        help="C_s. Default = 100",
                        dest="C_s")
    parser.add_argument('--C_f', default=1, type=float,
                        help="C_f. Default = 150",
                        dest="C_f")
    parser.add_argument('--h', default=1, type=float,
                        help="h. Default = 5",
                        dest="h")
    parser.add_argument('--b', default=38, type=int,
                        help="b. Default = 495",
                        dest="b")


    parser.add_argument('--max_episode_length', default=10000, type=int,
                        help="max_episode_length. Default = 100",
                        dest="max_episode_length")

    parser.add_argument('--warmup', default=0.1, type=float,
                        help="warmup. Default = 20",
                        dest="warmup")


    parser.add_argument('--setting', default=7, type=int,
                        help="setting. Default = 1",
                        dest="setting")

    parser.add_argument('--evalfrequency', default=10, type=int,
                        help="evalfrequency. Default = 1",
                        dest="evalfrequency")

    parser.add_argument('--parallel_evals', default=10, type=int,
                        help="parallel_evals. Default = 1",
                        dest="parallel_evals")

    parser.add_argument('--eval_length', default=100000, type=int,
                        help="eval_length. Default = 1",
                        dest="eval_length")

    parser.add_argument('--max_order', default=20, type=int,
                        help="max_order. Default = 5",
                        dest="max_order")
    parser.add_argument('--p', default=4, type=int,
                        help="p. Default = 5",
                        dest="p")
    parser.add_argument('--c', default=0, type=int,
                        help="c. Default = 0",
                        dest="c")

    args = parser.parse_args()

    #sample_paths = np.load('./sample_path_poisson_5.npy') Add any sample path you want to use for evaluation here.


    args.Demand_Max = 20

    demand_realizations = np.arange(args.Demand_Max + 1)
    demand_probabilities = poisson.pmf(np.arange(args.Demand_Max + 1), mu=5)
    demand_probabilities[-1] += 1 - np.sum(demand_probabilities)


    args.p = 4
    args.LT_min = 2
    args.LT_max = 2


    args.LT_s = args.LT_max
    LTs = np.load('LTs_%i_%i.npy' % (args.LT_min, args.LT_max))


    parameters = vars(args)

    while(True):
        objective(parameters)