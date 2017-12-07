
import sys
sys.path.append('.')
# Import global_var imperatively!
from simulator_reduced import SimulatorReduced, ScenarioLayout, simulations, global_var, simulation_test, Policy
import numpy as np
import copy
import matplotlib.pyplot as plt
import time


# TODO Find a behavior for beta => it has to decrease but at what speed?!
# TODO count waiting periods!
# TODO understand why the first decision in the test seems to be always the same...
# TODO Parallelize: without: 100 steps 37s

# To explore
#       Check behavior of the whole simulator
#       exploration proba...
#       SGDClassifier or MLP: depends on data inputs
#       Batch sizes, frequency of retraining
#       Parameters of your models
#       Design scenario on which your idea can work
#       Early stopping and overfitting problem
#

def main():
    # run simulations on n scenarios, m per scenario

    # Init
    start = time.time()
    print("** Initialization ** ")
    seed_test = 1071995  # so that test are always the same scenarios, set to -1 to avoid setting a seed
    beta = 1  # exploration half of the time at first
    n = 10  # nb of scenarios - nb of samples also at each retraining
    m = 10  # nb of simulations per scenario
    plot_test = True  # print trajectory of test number 1
    plot_nearest_neig = True
    plot_at_steps = [0, 200, 500, 700, 1000, 1500, 1999]

    steps = 2000  # nb of simulations
    test_step = 50
    nb_test_scenario = 10

    plot_scen = 3

    policy = Policy(policy="sgdclassifier")  # or "mlp" "sgdclassifier"
    policy_nearest_neighbor = Policy(policy="nearest_neighbor")

    sim = SimulatorReduced()
    sceLayout = ScenarioLayout(seed_test=seed_test, nb_test_scenario=nb_test_scenario)

    # scores
    scores_test = []

    # Comparison with nearest neighbor
    # Scenario nb 1 overview
    sceLayout.plot_scenario(jobs=sceLayout.test_scenarios[plot_scen])
    print(sceLayout.test_scenarios[plot_scen])

    print("Scenario 1: ")
    for i in sceLayout.test_scenarios[plot_scen].keys():
        print("     ", i)
        if i == 'pickup':
            for j in sceLayout.test_scenarios[plot_scen][i].keys():
                print("         ", j, ' : ', sceLayout.test_scenarios[plot_scen][i][j])
        else:
            for j in sceLayout.test_scenarios[plot_scen][i]:
                print("         ", j)

    average_test_nearest_neighbor = simulation_test(sim=sim,
                                                    scenarios=sceLayout.test_scenarios,
                                                    beta=0,
                                                    policy=policy_nearest_neighbor,
                                                    plot=plot_nearest_neig
                                                    )
    print("Average Reward Nearest Neighbor  : ", average_test_nearest_neighbor[0])
    print(average_test_nearest_neighbor[1])


    # Start Simulation
    for idx in range(0, steps):
        # modify beta or other parameters
        if 0 <= idx <= 800:
            beta = 0.6
        elif 801 <= idx <= 1200:
            beta = 0.5

        elif 1201 <= idx <= 1500:
            beta = 0.4
        else:
            beta = 0.3

        if idx % 50 == 0:
            print("Step ", idx)
        # Simulation
        # print(' ')
        # print("** Simulation ***")
        # print("...")
        data, labels = simulations(n=n,
                                   m=m,
                                   sim=sim,
                                   sceLayout=sceLayout,
                                   beta=beta,
                                   policy=policy)

        # Retrain
        # print(' ')
        # print("** Retraining **")
        # TODO PROBLEM We changed /Library/Frameworks/Python.framework/Versions/3.4/lib/python3.4/...
        # TODO ...site-packages/sklearn/neural_network/multilayer_perceptron.py
        # TODO line 920 => To be checked!
        policy.train_further(data=data, label=labels.astype(dtype=np.int8))
        """
        policy.classifier.fit(X=data, y=labels.astype(dtype=np.int8))
        """

        # Test
        if idx % test_step == 0:
            if idx in plot_at_steps and plot_test:
                plot_test_local = True
                sceLayout.plot_scenario(jobs=sceLayout.test_scenarios[plot_scen])

            else:
                plot_test_local = False
            print(" ")
            print("** Evaluation on Test Scenarios **")
            scenarios_test = copy.deepcopy(sceLayout.test_scenarios)
            mean_reward, rewards = simulation_test(sim=sim,
                                          scenarios=scenarios_test,
                                          beta=0,  # no exploration!
                                          policy=policy,
                                          plot=plot_test_local)

            print("Average Reward                   : ", mean_reward)
            print("Rewards Details                  : ", rewards)
            # if idx in plot_at_steps and plot_test:
            #     plt.show()

            scores_test.append(mean_reward)
        # Display trajectories

    end = time.time()

    # Visualization: Objective function evolution and traj
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(scores_test)
    # plt.plot(np.mean(scores_test, axis=1), 'r+')

    axes = fig.gca()
    axes.set_ylim([0, max(scores_test) +2])
    axes.set_xlim([0, steps/test_step])
    plt.show()

    print("Total time: ", end - start)


if __name__ == "__main__":
    main()
