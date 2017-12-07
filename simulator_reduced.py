import numpy as np
import matplotlib.pyplot as plt
from math import floor
from scipy.spatial.distance import cityblock as manh_dist
from scipy.spatial.distance import cdist
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier  # do not support incremental learning
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

import multiprocessing as mp
from multiprocessing import Process
import time
import copy

# TODO Verify the encoding of state data, it should only be made of 0.25, 0.5 and 0.75
# Problem with copy and deep copy etc.

# Global variables should be included as function parameter? Or global?
global_var = {
    "start_time": 0,
    "end_time"  : 1,
    "number_of_jobs": 30,  # total number of jobs (deliveries + pickups)
    "starting_point": (0, 0),  # tuple
    # "fixed_points": np.asarray([[3, 2], [5, 3], [3, 1], [7, 2], [6, 5], [4, 3], [3, 3]]),  # 2D numpy array (nb_deliveries, 2)
    # "fixed_points": np.asarray([[2, 1], [3, 2], [4, 3], [5, 4], [6, 3], [8, 2], [8, 4],  # ,
    #                             [2, 3], [3, 4], [4, 5]]),  # 2D numpy array (nb_deliveries, 2)  # 20.6 - 20.6
    "fixed_points": np.asarray([[2, 1], [4, 1], [6, 2], [7, 3], [8, 4], [9, 2], [8, 1]]),  # 19.1 -
    "duration_factor": 5,
    "grid_shape": (10, 10),
    "reward_pickup": 1,
    "reward_deliv": 1,
    "gaussian_sigma" : 2
}

#
#
#   Utilities
#
#


def visualize_grid(grid):
    """
    Visualization function for 2D grids
    :param grid: 2D np array
    :return:
    """
    plt.imshow(grid)
    # plt.colorbar()  # display color scale


def sample_2D(array, nb_sample, seed):
    """
    Sample 2D arrays
    :param array: 2D numpy array MUST BE a PROBA 2D ARRAY!
    :param nb_sample: int
    :param seed: int, set to -1 to avoid fixing a seed
    :return:
    """
    norm = np.sum(array)
    if norm < 0.99:
        raise ValueError('2D array does not represent a probability distribution:\n'
                         'np.sum(array) = '+str(norm)+'\n'
                         ' or is not 2D: \n'
                         'array.shape '+str(array.shape))

    array_local = np.asarray(array)
    choices = np.prod(array_local.shape)
    if seed != -1:
        np.random.seed(seed=seed)
    index = np.random.choice(choices, size=nb_sample, p=array_local.ravel(), replace=False)
    return np.unravel_index(index, dims=array_local.shape)


def filter_element_btw(time1, time2, dict_time):
    """
    Return datetime elements from list_time lower than time
    :param time1: any object that can be ordered with < > etc.
    :param time2: any object that can be ordered with < > etc.
    :param dict_time: dict with value being any object that can be ordered with < > etc.
    :return: keys from dict_time such that dict_time[key] in [time1, time2] if time1 <= time2 etc.
    """
    result = []
    if time1 < time2:
        for key, value in dict_time.items():
            if time1 <= value < time2:
                result.append(key)

    else:
        for key, value in dict_time.items():
            if time2 <= value <= time1:
                result.append(key)

    return result

########################################################################################################################

#
#
#   Classes
#
#


class ScenarioLayout:
    """
    Class allowing generating scenarios according to distributions and other parameters.
    Locations are placed among a grid.
    """
    def __init__(self, seed_test, nb_test_scenario=10):
        """

        :param seed_test: seed to fix tests scenarios /!\ not same as seed in generate_scenario
        function definition defintion
        """
        global global_var
        self.nb_test_scenario = nb_test_scenario
        self.grid_shape = global_var["grid_shape"]

        # Distributions
        self.distrib_appearance_grid = self.initialize_distrib_appearance()
        self.distrib_time_mean_grid = self.initialize_distrib_time_mean()

        self.distrib_time_var_grid = self.initialize_distrib_time_var()

        # Known points (deliveries) - no need for deep copies as self.deliveries is not modified anywhere
        self.deliveries = global_var["fixed_points"]

        # Test scenarios for learning evaluation over time
        self.test_scenarios = [
            self.generate_scenario(seed=seed_test+i) for _, i in zip(range(0, self.nb_test_scenario),
                                                                   range(0, self.nb_test_scenario))
        ]

    def __repr__(self):
        return "Scenario Overview: \n" \
               "    Deliveries:         : %s \n" \
               "    Pickups" % (self.deliveries)

    def gaussian_peak_2D(self, gaussian_position):
        """

        :param gaussian_position: tuple, (x-position, y-position)
        :return: grid
        """
        # création de deux rampes "ouvertes" (en opposition avec la caractère "dense" des mgrid) d'indices
        M, N = np.ogrid[0:self.grid_shape[0], 0:self.grid_shape[1]]

        # projection de la position du pic de la gaussienne
        X,Y = M - gaussian_position[1], N - gaussian_position[0]
        # calcul du noyau gaussien centré
        kernel = np.exp(-(X ** 2 + Y ** 2) / (2 * global_var["gaussian_sigma"] ** 2))
        return kernel

    # Distribution initialization
    def initialize_distrib_appearance(self):
        """
        Distribution matrix of likelyness of new pickup occuring
        :return: 2D np array
        """
        # uniform
        # return 1/(self.grid_shape[0]*self.grid_shape[1])*np.ones(self.grid_shape)

        # gaussian peaks following a curve:
        # curve y = -x^2 + 2*x
        x_centers = np.arange(0, self.grid_shape[1], 1)
        y_centers = np.floor(-1/(self.grid_shape[1]*self.grid_shape[0]**2) * x_centers**2 + \
                             x_centers * 2/(self.grid_shape[1]*self.grid_shape[0]))
        # initialize grid
        grid = np.zeros(self.grid_shape)

        # add peaks
        # TODO Add robustness in controlling the position of the high probability parts!
        for i in range(floor(x_centers.shape[0]*1/4), floor(x_centers.shape[0]*3/4)):
            grid += self.gaussian_peak_2D((x_centers[-i]+2, y_centers[-i]))

        # normalization (probability distrib)
        return grid/grid.sum()

    def initialize_distrib_time_mean(self):
        """
        Distribution matrix of the appearance mean time of each location
        :return: 2D np array
        """
        # mid-morning for all
        # return 0.5*np.ones(self.grid_shape)

        # up the diagonal, rest 0
        grid = np.copy(self.distrib_appearance_grid)
        flag_appearance_grid = self.distrib_appearance_grid > np.amax(self.distrib_appearance_grid)/10
        grid[flag_appearance_grid] = 0.3

        return grid

    def initialize_distrib_time_var(self):
        """
        Distribution matrix of the appearance variance time of each location
        :return: 2D np array
        """
        # TODO adjust this
        return 0.3*np.ones(self.grid_shape)

    # Generate scenario
    def generate_scenario(self, seed):
        """
        Sample pickups and appearance time and wrap this with fixed known points
        :param seed: set to -1 if you don't wanna control the seed and make sure more true random
        scenarios are generated (...)
        :return: dict, {'delivery': [[x-coord, y-coord],...],
                        'pickup'  : {[x-coord, y-coord] : appearanceTime,
                                      ...}
                                    }
        """
        global global_var
        # Sample pickup locations
        y_p, x_p = sample_2D(array=self.distrib_appearance_grid,
                             nb_sample=global_var["number_of_jobs"] - global_var["fixed_points"].shape[0],
                             seed=seed)

        # create dictionary to be returned (tuple format)
        jobs = dict()
        jobs["delivery"] = [(global_var["fixed_points"][i, 0], global_var["fixed_points"][i, 1])
                            for i in range(0, global_var["fixed_points"].shape[0])]

        # sample time of appearance of pickups
        # Normal distribution
        jobs["pickup"] = {
            (y, x): min(1, max(0.0001, np.random.normal(loc=self.distrib_time_mean_grid[y, x],
                                     scale=self.distrib_time_var_grid[y, x])))
            for x, y in zip(x_p, y_p)
        }
        return jobs

    def plot_scenario(self, jobs):
        """
        Plot deliveries and pickups on top of appearance distrib
        :param jobs: dict, a scenario from self.generate_scenario
        :return:
        """
        visualize_grid(self.distrib_appearance_grid)

        # deliveries
        coords_deliveries = np.asarray(jobs["delivery"])
        plt.scatter(coords_deliveries[:, 1], coords_deliveries[:, 0], color='g', marker='+')

        # pickups
        coords_pickups = np.asarray(list(jobs["pickup"].keys()))
        plt.scatter(coords_pickups[:, 1], coords_pickups[:, 0], color='r', marker='*')


class Policy:
    """
    Create a policy and its exploration version, as well as tools to retrain it.
    One such object must be created for the whole training (or several?)
    """
    def __init__(self, policy="nn"):
        """

        :param policy: string, 'nn' pour neural net, or 'nearest_neighbor'
        """
        global global_var
        self.n_features = np.prod(global_var["grid_shape"]) + 1  # all locations + time
        self.n_classes = self.n_features - 1
        self.policy = policy

        # neural network or anything => SGDClassifier because can be retrained
        # TODO choose learning rate
        if self.policy == 'sgdclassifier':
            self.classifier = SGDClassifier(loss="log",  # log or modified_huber
                                            penalty="elasticnet",
                                            alpha=0.0001,
                                            fit_intercept=True,
                                            shuffle=False,
                                            warm_start=True,  # doesn't erase result of previous call to .fit
                                            n_jobs=-1
                                            )
        elif self.policy == 'passiveaggressive':
            self.classifier = PassiveAggressiveClassifier(loss='hinge',
                                                          fit_intercept=True,  # data not centered
                                                          n_jobs=-1  # to use all CPU
                                                          )
        elif self.policy == 'mlp':
            self.classifier = MLPClassifier(hidden_layer_sizes=(200,50),
                                            activation="logistic",  # default; relu
                                            solver="adam",  # default
                                            alpha=0.0001,  # default
                                            # batch_size=min(200, n_samples),  # default
                                            learning_rate="constant",  # default
                                            learning_rate_init=0.001, # default
                                            max_iter=200,  # default
                                            shuffle=True,  # default
                                            warm_start=True)

        # optional?
        # self.classifier.classes_ = np.arange(start=0, stop=self.n_classes)

        # scaler => maybe not necessary as time is in [0,1] and location values will be chosen in [0,1] too
        # scaler = StandardScaler()

        # Initial training on dummy data that contains all labels!
        if self.policy != "nearest_neighbor":

            self.classifier.partial_fit(X=np.random.rand(self.n_classes, self.n_features),
                                        y=np.arange(0, self.n_classes, step=1),  # to make sure all classes are present!
                                        classes=np.arange(start=0, stop=self.n_classes)
                                        )


            """
            self.classifier.fit(X=np.random.rand(self.n_classes, self.n_features),
                                y=np.arange(0, self.n_classes))  # to make sure all classes are present!)
            """

    def make_decision(self, state, beta):
        """

        :param state: State object, to be transformed as features
        :param beta: float in [0,1], probability with which a random decision is taken
        :param mode: string, 'exploration' or 'exploitation'
        :return: decision, tuple (location on grid)
            /!\ We return -1 when no more jobs to be done => waiting period
        """
        # if waiting period, return now to avoid useless computations
        if len(state.D_k) == 0 and not any(state.P_k):
            return -1

        if self.policy != 'nearest_neighbor':
            # encode state on features
            X = state.encode_state()

            # Identify jobs we can select
            if len(state.D_k) != 0:  # non empty list
                D_k = np.ravel_multi_index(np.transpose(np.asarray(state.D_k, dtype=np.int8)),
                                           dims=global_var["grid_shape"]
                                           )
            else:
                D_k = np.asarray([], dtype=np.int8)

            if any(state.P_k):  # non empty dict
                P_k = np.ravel_multi_index(np.transpose(np.asarray(list(state.P_k.keys()))),
                                           dims=global_var["grid_shape"]
                                           )
            else:
                P_k = np.array([])

            D_k_and_P_k = np.concatenate((D_k, P_k))
            # indices of jobs to choose from
            D_k_and_P_k = D_k_and_P_k.astype(np.int8)

            if np.random.uniform() < beta:
                # print("exploring...")
                # random among possible location
                decision = np.random.choice(D_k_and_P_k, size=1)
                # decision = random.choice(list(state.P_k.keys())+state.D_k)
                return np.unravel_index(decision, dims=global_var["grid_shape"])

            else:
                # print("exploiting...")
                # NN evaluation
                # decision = self.classifier.predict(X)
                decision = self.classifier.predict_proba(X)

                # Raise score of possible jobs so that taking the max eliminate other non available locations
                decision[0, D_k_and_P_k] += 2
                decision_among_available_jobs = np.argsort(decision[0, :])[-1]  # max proba

                # unravel: flat index to coordinate tuple
                return np.unravel_index(decision_among_available_jobs, dims=global_var["grid_shape"])

        elif self.policy == 'nearest_neighbor':
            # Distance from current location state.c_k from state.D_k and state.P_k
            curr_loc = np.asarray([state.c_k])
            # np.reshape(curr_loc, newshape=(1, 2))

            # Remaining jobs coordinates
            jobs_deliv = np.array(state.D_k)
            pickups = list(state.P_k.keys())
            jobs_pickup = np.array(pickups)

            # print("shape of curr loc ", jobs_pickup.shape)
            # distance between curr_loc and jobs
            if jobs_deliv.size != 0:
                dist_deliv = cdist(curr_loc, jobs_deliv, metric='cityblock')
                min_deliv = dist_deliv.min()
                argmin_deliv = dist_deliv.argmin()
            else:
                min_deliv = 2 * np.prod(
                    global_var["grid_shape"])  # high number to make sure it's not the possible nearest

            if jobs_pickup.size != 0:
                dist_pickup = cdist(curr_loc, jobs_pickup, metric='cityblock')
                min_pickup = dist_pickup.min()
                argmin_pickup = dist_pickup.argmin()
            else:
                min_pickup = 2 * np.prod(global_var["grid_shape"])  #

            if jobs_pickup.size == 0 and jobs_deliv.size == 0:
                return -1

            if min_pickup < min_deliv:
                return pickups[argmin_pickup]
            else:
                return state.D_k[argmin_deliv]

    def train_further(self, data, label):
        """

        :param data: np array (n_samples, n_features)
        :param label: np array (n_samples,)
        :return: updated policy
        """
        """
        self.classifier.partial_fit(X=data, y=label)
        """
        self.classifier.partial_fit(X=data, y=label)

# add the global policy to the global_var dict
# global_var["policy"] = Policy()

class State():
    """
    Contains description of current state
    """
    def __init__(self, scenario, starting_time=0):
        """

        :param starting_time: int
        :param scenario: dict from ScenarioLayout.generate_scenario( )
        """
        self.scenario = scenario

        # Index
        self.k = 0

        # current time
        self.t_k = starting_time

        # current location
        self.c_k = copy.deepcopy(global_var["starting_point"])  # nested structure that will be modified => deepcopy

        # remaining deliveries - list of coords (tuples)
        self.D_k = copy.deepcopy(self.scenario["delivery"])  # nested structure that will be modified => deepcopy

        # remaining pickups - empty at first
        # TODO maybe not empty in future case
        self.P_k = dict()

    def __repr__(self):
        """String displayed when # printing"""
        return "State at step "+str(self.k)+"\n" \
               "    - Current time         : "+str(self.t_k)+"\n"\
               "    - Current location {}  : "+str( self.c_k)+"\n"\
               "    - Remaining deliveries : "+str(self.D_k)+"\n"\
               "    - Remaining pickups    : "+str(self.P_k)

    def encode_state(self):
        """
        Return encoded version of self.
        That is:
            numpy array of np.prod(global_var["grid_shape"]) + 1
            time, one-hot-encoding
        :return: np array (1, nb_features)
        """
        # encoded vector
        encoded_state = np.zeros(shape=(1, np.prod(global_var["grid_shape"]) + 1))

        # time
        encoded_state[0, 0] = self.t_k

        # encode deliveries
        # TODO ravel_multi_index doesn't seem to do what we want
        D_k = np.transpose(np.asarray(self.D_k))
        P_k = np.transpose(np.asarray(list(self.P_k.keys())))

        if D_k.size != 0:
            # +1 because first index is devoted to time
            one_hot_deliv = np.ravel_multi_index(D_k, dims=global_var["grid_shape"])+1
            encoded_state[0, one_hot_deliv] = 0.75

        if P_k.size != 0:
            # encode pickups
            # +1 because first index is devoted to time
            one_hot_pickup = np.ravel_multi_index(P_k, dims=global_var["grid_shape"])+1
            encoded_state[0, one_hot_pickup] = 0.25

        # current location encoding
        one_hot_current = np.ravel_multi_index(self.c_k, dims=global_var["grid_shape"])+1
        encoded_state[0, one_hot_current] = 0.5

        return encoded_state


class SimulatorReduced():
    """
    One simulator is gonna be created per scenario, using one policy.
    """
    def __init__(self):
        """

        """
        # TODO define State( ) and add arguments here from scenario and starting time
        # initial_state = State()

    def get_travel_time(self, start_loc, end_loc):
        """
        Travel time is a function of distance. Here distance is Manhattan distance
        :param start_loc: int, in [0, np.prod(grid_shape)]
        :param end_loc: int, in [0, np.prod(grid_shape)]
        :return:
        """
        # Manhattan distance
        dist = manh_dist(start_loc, end_loc)

        # Apply function from distance to travel time
        # TODO max_path_length represents the longest distance possible,
        # TODO taking 1/global_var["duration_factor"] time

        max_path_length = np.sum(global_var["grid_shape"])
        return (-1/(global_var["duration_factor"]*max_path_length**2)*dist**2 +\
               dist*2/(max_path_length*global_var["duration_factor"]))*1.5

    def get_service_time(self, loc):
        """
        Service time for a location
        :param loc, int in [0, np.prod(grid_shape)]
        :return:
        """
        # For now constant, fraction of total available time
        constant = 0.008  # 0.8% is the share of time spent on servicing a job (2min for 3h30 of operations)
        return constant

    def generate_next_state(self, decision, state, scenario):
        """

        :param decision: tuple (coord1, coord2)
        :param state: State object
        :param scenario: ScenarioLayout object, used only to get new appearing jobs
        :return: tuple, (state, reward) updated
        """
        # Waiting period => same current location, time increases and jobs are being updated
        if decision == -1:
            # waiting time
            waiting_time = 1/120

            # current location doesn't change

            # add new jobs appearing between state.t_k and new time from scenario
            new_jobs_locations = filter_element_btw(state.t_k, state.t_k + waiting_time, scenario['pickup'])
            for job in new_jobs_locations:
                state.P_k[job] = scenario["pickup"][job]

            # update time
            state.t_k += waiting_time

            # update step
            state.k += 1

            return state, 0

        else:
            reward = 0
            # compute next time but don't change the state yet!
            travel_time = self.get_travel_time(state.c_k, decision)
            # print("time to go ", travel_time)

            # transform decision into a normal looking tuple instead of (array([8]), ...)
            decision = int(decision[0]), int(decision[1])

            # current location becomes the decision
            state.c_k = decision
            # Remove current location from jobs
            if decision in state.D_k:  # decision is a delivery
                state.D_k.remove(decision)
                reward = global_var["reward_deliv"]
            elif decision in list(state.P_k):
                state.P_k.pop(decision)
                reward = global_var["reward_pickup"]

            else:
                raise ValueError('decision '+str(decision)+' is not among deliveries or pickups')

            # add new jobs appearing between state.t_k and new time from scenario
            new_jobs_locations = filter_element_btw(state.t_k, state.t_k + travel_time, scenario['pickup'])
            for job in new_jobs_locations:
                # TODO Check, normally no deepcopy is needed here.
                state.P_k[job] = scenario["pickup"][job]

            # update time
            state.t_k += travel_time

            # update step
            state.k += 1

            return state, reward

    def run_simulation(self, state, scenario, beta, policy, plot=False):
        """
        Run One simulation on one scenario
        :param state: State object
        :param scenario: dict from ScenarioLayout.generate_scenario( )
        :param beta: float in [0,1] proba of exploring
        :param policy: Policy object or string equals to nearest_neighbor
        :return: tuple, (total reward of the trajectory, history of states)
        """
        # To be returned at the end
        total_reward = 0
        state_chronology = state.encode_state()
        path = [(0, 0)]  # list of current locations
        # While there is still time and jobs
        while state.t_k < global_var["end_time"]:  # and (len(state.P_k) != 0 or any(state.D_k)):
            # print("*********")
            # print(state)
            # Make a decision
            # print("-- Making a decision --")

            decision = policy.make_decision(state, beta=beta)
            # print("     decision= ", decision)

            # Generate new state
            # print("-- Finding New State --")
            state, reward = self.generate_next_state(decision=decision, state=state, scenario=scenario)
            # print("     reward= ", reward)

            # Register what's happening
            if decision != -1:  # we don't register waiting periods
                total_reward += reward
                state_chronology = np.concatenate((state_chronology, state.encode_state()), axis=0)
                path.append(state.c_k)

        if plot:
            plt.plot(np.asarray(path)[:, 1], np.asarray(path)[:, 0])
            plt.show()
            # now = time.time()
            # plt.savefig("traj_"+str(int(now))+".png")
            plt.clf()  # clear figure

        return total_reward, state_chronology, path


def simulations(n, m, sim, sceLayout, beta, policy):
    """
    Use n scenarios to run m simulation against each. Return the bunch of new data obtained by selecting the
    best runs from each of the scenario m simulations.

    :param n: int, nb of scenarios
    :param m: int, nb of simulations per scenario
    :param sim: SimulatorReduced object
    :param sceLayout: ScenarioLayout object
    :param policy: Policy object
    :return:
    """
    # TODO We do loop for now but will have to use parallel programming!!
    # set the seed to -1 to remove any seed setting


    # store data for future retraining of the policy. Shape (n, global_variable["grid_shape"])
    retrain_data = np.empty(shape=(0, np.prod(global_var["grid_shape"])+1))
    retrain_labels = np.empty(shape=(0,))

    for nb_scenario in range(0, n):  # TODO this can be parallelized
        # --------------------------------------------------------------------------------------
        seed_scenario = -1  # easy way of fixing scenarios for test purpose OR -1

        # Scenarios creation
        sce1 = sceLayout.generate_scenario(seed=seed_scenario)
        # print("Scenario ", nb_scenario, ' : ')
        # print(sce1)
        # sceLayout.plot_scenario(sce1)

        # print("SCENARIO ")
        # print(sce1)
        # print("*_-_-_-_-_-_-_-_*")

        # m simulations and their results
        record_rewards = []
        record_data = []
        record_path = []

        # Parallel Version
        """
        # SOLUTION 1  =>  Don't know how to retrieve output+doesn't seem to work...
        q = mp.Queue()
        # jobs
        jobs = [Process(target=parallel_simulation,
                        args=(sim,
                              sce1,
                              beta,
                              policy))
                for _ in range(0, m)
                ]

        # start processes
        for i in jobs:
            i.start()

        # join
        for j in jobs:
            j.join()

        # retrieve results
        for k in jobs:
            print(k.get())

        # SOLUTION 2  =>  Overwhelm the computer...
        pool = mp.Pool(processes=4)
        results = [pool.apply(parallel_scenario_simulation, args=(sim, sce1, beta, policy)) for _ in range(0, m)]
        # output = [p.get() for p in results]

        for result in results:
            record_rewards.append(result[0])
            record_data.append(result[1])
            record_path.append(result[2])
        """

        # Non parallel version

        for _ in range(0, m):  # TODO can probably be parallelized too
            # --------------------------------------------------------------------------------------
            # input: sce1, beta, policy
            # output: append to record_data, record_rewards, record_path

            # print("NEW SIM: ", sce1)
            # Copy of the global scenario
            sce_local = copy.deepcopy(sce1)

            # initialize state
            state_local = State(scenario=sce_local)

            rew, history, path = sim.run_simulation(state=state_local,
                                                    scenario=sce_local,
                                                    beta=beta,
                                                    policy=policy
                                                    )
            # # print(history)
            record_data.append(history)
            record_rewards.append(rew)
            record_path.append(path)
            # --------------------------------------------------------------------------------------

        # position of max
        max_sim = np.asarray(record_rewards).argmax()
        # data of max
        max_data = np.asarray(record_data[max_sim])
        # print("Max Path ", record_path[max_sim])
        # print("best rewards ", record_rewards[max_sim])
        # Trajectory with the highest score
        traj = np.ravel_multi_index(np.transpose(record_path[max_sim]),dims=global_var["grid_shape"])
        # print("Max rewards obtained ", record_rewards[max_sim])
        # Very important Part /!\ Careful with removing last line of retrain_data and first line of traj
        retrain_data = np.concatenate((retrain_data, max_data[:-1, :]))
        retrain_labels = np.concatenate((retrain_labels, traj[1:]))
        # --------------------------------------------------------------------------------------
    return retrain_data, retrain_labels

def simulation_test(sim, scenarios, beta, policy, plot=False):
    """

    :param sim: SimulatorReduced object, contains the policy most importantly
    :param scenarios: list of scenarios, so list of dicts
    :param policy: Policy object
    :return: float, average reward obtained when serving the scenarios
    """
    rewards = []
    for scenario, nb in zip(scenarios, range(0, len(scenarios))):
        # initialize state
        state_local = State(scenario=scenario)

        if nb == 3 and plot == True:
            plot_local = True
        else:
            plot_local = False
        # simulation
        rew, history, path = sim.run_simulation(state=state_local,
                                                scenario=scenario,
                                                beta=beta,
                                                policy=policy,
                                                plot=plot_local)
        # print("Path: ")
        # print(path)
        rewards.append(rew)
    # return rewards
    return sum(rewards)/len(rewards), rewards  # mean of scores

######################################
#
#   Parallel functions
#
######################################

def parallel_simulation(sim, sce, beta, policy):
    sce_local = copy.deepcopy(sce)

    # initialize state
    state_local = State(scenario=sce_local)

    rew, history, path = sim.run_simulation(state=state_local,
                                                 scenario=sce_local,
                                                 beta=beta,
                                                 policy=policy
                                            )
    return rew, history, path

if __name__ == '__main__':
    # Init
    seed_test = 1071995  # so that test are always the same scenarios, set to -1 to avoid setting a seed
    policy = Policy( policy="sgdclassifier")
    pol_nearest = Policy(policy="nearest_neighbor")
    sim = SimulatorReduced()
    sceLayout = ScenarioLayout(seed_test=seed_test)
    beta = 0.3

    # Sim
    # sceLayout.plot_scenario(jobs=sceLayout.test_scenarios[1])
    grid = sceLayout.distrib_time_var_grid
    visualize_grid(grid)
    # data, labels = simulations(n=1, m=1, sim=sim, sceLayout=sceLayout, beta=beta, policy=policy)
    # data, labels = simulations(n=1, m=1, sim=sim, sceLayout=sceLayout, beta=beta, policy=pol_nearest)
    # rwardsS = simulation_test(sim=sim, scenarios=sceLayout.test_scenarios, beta=beta, policy=policy)
    # rwards = simulation_test(sim=sim, scenarios=sceLayout.test_scenarios, beta=beta, policy=pol_nearest)
    # print("rewardss ", rwardsS)
    # print("rewards ", rwards)
    plt.show()



