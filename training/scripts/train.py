#!/usr/bin/env python
import sys, rospy, time
from datetime import datetime

import torch
import numpy as np

from std_msgs.msg import Empty

from training.scripts.PPO import PPO_custom
from rl_utils.envs.flatland_gym_env import (
    FlatlandEnv,
)

from tools.argsparser import parse_training_args
from tools.general import *



from tools.env_utils import init_envs


def on_shutdown(env):
    env.close()
    sys.exit()


def main():
    ##################load config and ros setting####################
    args, _ = parse_training_args()

    config = load_config(args.config)

    populate_ros_configs(config)

    # in debug mode, we emulate multiprocessing on only one process
    # in order to be better able to locate bugs
    if config["debug_mode"]:
        rospy.init_node("debug_node", disable_signals=False)

    # generate agent name and model specific paths
    generate_agent_name(config)
    PATHS = get_paths(config)

    print("________ STARTING TRAINING WITH:  %s ________\n" % config["agent_name"])

    # for training with start_arena_flatland.launch
    ns_for_nodes = rospy.get_param("/ns_for_nodes", True)

    # check if simulations are booted
    wait_for_nodes(with_ns=ns_for_nodes, n_envs=config["n_envs"], timeout=5)

    # initialize hyperparameters (save to/ load from json)
    config = initialize_config(
        PATHS=PATHS,
        config=config,
        n_envs=config["n_envs"],
        debug_mode=config["debug_mode"],
    )
    populate_ros_params(config)
    ###################################################################
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    has_continuous_action_space = not config["rl_agent"]["discrete_action_space"]  # continuous action space; else discrete

    max_ep_len = config["max_num_moves_per_eps"]                   # max timesteps in one episode
    max_training_timesteps = config["n_timesteps"]   # break training loop if timeteps > max_training_timesteps

    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################


    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    #lr_actor = config["rl_agent"]["ppo"]["learning_rate"]       # learning rate for actor network
    #lr_critic = config["rl_agent"]["ppo"]["learning_rate_critic"]       # learning rate for critic network
    lr_actor = 1e-4
    lr_critic = 1e-4
    lr_commonLayer = 1e-4
    lr_policy = 1e-4
    lr_value = 1e-4









    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    ################ setting up Flatland env ################
    curriculum_config = config["callbacks"]["training_curriculum"]
    env = FlatlandEnv(
                "sim_1",
                config["rl_agent"]["reward_fnc"],
                config["rl_agent"]["discrete_action_space"],
                goal_radius=config["goal_radius"],
                max_steps_per_episode=config["max_num_moves_per_eps"],
                task_mode=config["task_mode"],
                curr_stage=curriculum_config["curr_stage"],
                PATHS=PATHS,
            )



    #env, eval_env = init_envs(config, PATHS, ns_for_nodes)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    #########################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder
    checkpoint_path = PATHS["model"] + "PPO_{}_{}_{}.pth".format(rospy.get_param("robot_model"), random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO_custom(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    rospy.on_shutdown(lambda: on_shutdown(env))

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
           
            state, reward, done, _ = env.step(action)
            

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        i_episode += 1

    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

    # Send Task finished to Backend
    if rospy.get_param("/is_webapp_docker", False):
        publisher = rospy.Publisher("training_finished", Empty, queue_size=10)
        
        while publisher.get_num_connections() <= 0:
            pass

        publisher.publish(Empty())

    sys.exit()


if __name__ == "__main__":
    main()