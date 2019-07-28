from subprocess import run
import numpy as np
import os
import datetime

subsystem_number = [8]
network_nodes = [2048]
epsilon_decay = [0.99995]
batch_sizes = [32]
epochs = [75]
memory = [75000]
sl_epochs = [5]
number_of_training_runs = 6
learning_rates = [np.exp(-6.5)]

save_path = '/upb/departments/pc2/groups/hpc-prf-acg/results/'

idx = 1
for n, k, eps, epoch, mem, sl_epoch, batch_size in zip(subsystem_number, network_nodes, epsilon_decay, epochs, memory,
                                                       sl_epochs, batch_sizes):
    start = datetime.datetime.now()
    sys_path = os.path.join(save_path, 'S' + str(n) + '_' + start.strftime("%I%M%p%B%d%Y") + '/')
    for lr in learning_rates:
        lr_path = os.path.join(sys_path, 'lr_' + str(lr) + '/')
        for j in range(number_of_training_runs):
            run_path = os.path.join(lr_path, 'run_' + str(j) + '/')
            run("ccsalloc ~/tensorflow.sh /scratch/hpc-prf-acg/aredder/anaconda3/envs/tf_gpu/bin/python "
                "/upb/scratch/departments/pc2/groups/hpc-prf-acg/aredder/Projects/pb_cown/rl_agents"
                "/agent_comparison.py True " + run_path + " " + str(n) + " " + str(k) + " " + str(lr)
                + " " + str(eps) + " " + str(epoch) + " " + str(mem) + " " + str(sl_epoch) + " " + str(batch_size),
                shell=True)

