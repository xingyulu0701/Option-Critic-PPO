from tensorboardX import SummaryWriter
import copy
import glob
import os
import time
import gtimer as gt 
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy, OptionCritic
from storage import RolloutStorage
from visualize import visdom_plot

from grid_world_env import get_test_env
args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

writer = SummaryWriter(log_dir=args.log_dir)

num_options = 2

def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    # envs = [make_env(args.env_name, args.seed, i, args.log_dir) for i in range(args.num_processes)]
    # env = get_test_env("001")
    envs = [lambda: get_test_env("000") for _ in range(args.num_processes)]
    # num_states = len(env.all_possible_states())
    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if len(envs.observation_space.shape) == 3:
        actor_critic = OptionCritic(num_options, obs_shape[0], envs.action_space, args.recurrent_policy)
    else:
        # assert not args.recurrent_policy, \
        #     "Recurrent policy is not implemented for the MLP controller"
        # actor_critic = MLPPolicy(obs_shape[0], envs.action_space)
        raise NotImplementedError()

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        # optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
        raise NotImplementedError()
    elif args.algo == 'ppo':
        optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps = args.eps)
    elif args.algo == 'acktr':
        # optimizer = KFACOptimizer(actor_critic)
        raise NotImplementedError()

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size, num_options)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)
    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])
    optionSelection = 0
    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()
    start = time.time()
    #print(options)
    #print(options[0])
    for j in range(num_updates):
        options = [-1] * args.num_processes
        for step in range(args.num_steps):
            # Choose Option 
            t0 = time.time()
            selection_value, new_option, option_log_prob, states = actor_critic.get_option(Variable(rollouts.observations[step], volatile=True),
                Variable(rollouts.states[step], volatile=True),
                Variable(rollouts.masks[step], volatile=True))
                   # print(new_option)
            for i in range(args.num_processes):
                if options[i] == -1:
                    options[i] = new_option[i].data[0]
            #print("option is:")
            #print(options)
            t1 = time.time()
            # Sample actions
            value, action, action_log_prob, states = actor_critic.get_output(
                    options,
                    Variable(rollouts.observations[step], volatile=True),
                    Variable(rollouts.states[step], volatile=True),
                    Variable(rollouts.masks[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()
            t2 = time.time()
            # Termination 
            term_value, termination, termination_log_prob, _ = actor_critic.get_termination(
                options,
                Variable(rollouts.observations[step], volatile=True),
                    Variable(rollouts.states[step], volatile=True),
                    Variable(rollouts.masks[step], volatile=True))
            termination = torch.LongTensor([termination[i].data[0] for i in range(termination.shape[0])])
            t3 = time.time()
            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # newIndex = obs_to_int(obs)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks


            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks
            update_current_obs(obs)
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, reward, masks, options, termination)
            
            for i in range(termination.shape[0]):
                if termination[i] == 1:
                    options[i] = -1
            t4 = time.time()
            #print("part1")
            #print(t1 - t0)
            #print("part2")
            #print(t2-t1)
            #print("part3")
            #print(t3-t2)
            #print("part4")
            #print(t4-t3)
        for i in range(args.num_processes):
            if options[i]== -1:
                selection_value, new_option, option_log_prob, states = actor_critic.get_option(Variable(rollouts.observations[step], volatile=True),
                    Variable(rollouts.states[step], volatile=True),
                    Variable(rollouts.masks[step], volatile=True))
                # print(new_option)
            options[i] = new_option[i].data[0]
        rollouts.options[step+1].copy_(torch.LongTensor(options))
        next_value = actor_critic.get_output(options,Variable(rollouts.observations[-1], volatile=True),
                                  Variable(rollouts.states[-1], volatile=True),
                                  Variable(rollouts.masks[-1], volatile=True))[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        if args.algo in ['a2c', 'acktr']:
            raise NotImplementedError()
        elif args.algo == 'ppo':
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            for e in range(args.ppo_epoch):
                for i in range(args.num_steps):
                    # Get the ith step during exploration
                    options = rollouts.options[i]
                    #print(options)
                    adv_targ = Variable(advantages[i])
                    old_action_log_probs = rollouts.action_log_probs[i]
                    termination = rollouts.optionSelection[i]
                    #print(termination)
                    # Use critic value of option nn to update option parameters
                    values, action_log_probs, dist_entropy, states = actor_critic.evaluate_option(
                        Variable(rollouts.observations[i]),
                        Variable(rollouts.states[i]),
                        Variable(rollouts.masks[i]),
                        Variable(rollouts.actions[i]), options)
                    #print(action_log_probs)
                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)
                    value_loss = (Variable(rollouts.returns[i]) - values).pow(2).mean()

                    selection_log_prob = actor_critic.evaluate_selection(
                        Variable(rollouts.observations[i]),
                        Variable(rollouts.states[i]),
                        Variable(rollouts.masks[i]),
                        Variable(termination),
                        Variable(rollouts.options[i].type(torch.cuda.LongTensor)))
                    V_Omega = selection_log_prob * values 

                    # Update termination parameters 
                    termination_log_prob = actor_critic.evaluate_termination(
                        Variable(rollouts.observations[i]),
                        Variable(rollouts.states[i]),
                        Variable(rollouts.masks[i]),
                        Variable(termination.type(torch.cuda.LongTensor)),
                        rollouts.options[i+1])
                    left_values = []
                    right_values = []
                    for i in range(args.num_processes):
                        if int(termination[i]) == 1:
                            left_values.append(V_Omega[i])
                            right_values.append(values[i])
                        elif int(termination[i]) == 0:
                            left_values.append(values[i])
                            right_values.append(V_Omega[i])
                    left_values = torch.cat(left_values)
                    right_values = torch.cat(right_values)
                    termination_loss = (- torch.exp(termination_log_prob) * left_values - (1 - torch.exp(termination_log_prob)) * right_values).mean()
                    optimizer.zero_grad()

                    (action_loss + value_loss+ termination_loss - V_Omega.mean()).backward()
                    nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))
            writer.add_scaler("data/final_reward", final_rewards.max())

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                print("hit")
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass

if __name__ == "__main__":
    main()
