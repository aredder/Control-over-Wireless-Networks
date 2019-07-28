import numpy as np
import scipy as sp
from system_env import system_models


def main():
    system_number = [16]
    goal_loss = 1

    for x in system_number:
        system_hp = {'system_dimension': 2*x,
                     'number_of_subsystems': x,
                     'dependency': 0.1,
                     'ratio_of_stable_subsystems': 0.5,
                     'init_cov_scale': 1,
                     'noise_cov_scale': 0.1,
                     'state_cost_scale': 1,
                     'control_cost_scale': 0.1}

        current_loss = 0
        best_seed = 0
        for y in range(0, 500):
            system = system_models.LinearSystem(dimension=system_hp['system_dimension'],
                                                subsystems=system_hp['number_of_subsystems'],
                                                init_cov=np.eye(system_hp['system_dimension']) *
                                                system_hp['init_cov_scale'],
                                                noise_cov=np.eye(system_hp['system_dimension']) *
                                                system_hp['noise_cov_scale'],
                                                dependency=system_hp['dependency'],
                                                stability=system_hp['ratio_of_stable_subsystems'], optional_seed=y)
            system.q_system = np.eye(np.shape(system.A)[0]) * system_hp['state_cost_scale']
            system.r_system = np.eye(np.shape(system.B)[1]) * system_hp['control_cost_scale']

            optimal_avg_loss = np.trace(sp.linalg.solve_discrete_are(system.A, system.B, system.q_system,
                                                                     system.r_system)
                                        @ np.eye(system_hp['system_dimension']) * system_hp['noise_cov_scale'])

            if np.abs(goal_loss-optimal_avg_loss / system_hp['number_of_subsystems']) - \
                    np.abs(goal_loss-current_loss) < 0:
                current_loss = optimal_avg_loss / system_hp['number_of_subsystems']
                best_seed = y
                print(current_loss)
                print(y)

        print('Number of subsystems: ' + str(x))
        print('Seed: ' + str(best_seed))
        print('Corresponding loss: ' + str(current_loss))


if __name__ == "__main__":
    main()
