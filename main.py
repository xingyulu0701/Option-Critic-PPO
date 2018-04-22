import copy
import glob
import os
import time

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

    envs = [make_env(args.env_name, args.seed, i, args.log_dir)
                for i in range(args.num_processes)]
    # env = get_test_env("001")

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
        optionOptimizers = [optim.Adam(actor_critic.intraOption[i].parameters(), args.lr, eps=args.eps) for i in range(num_options)]
        terminationOptimizers = [optim.Adam(actor_critic.terminationOption[i].parameters(), args.lr, eps=args.eps) for i in range(num_options)]
        optionSelectionOptimizer = optim.Adam(actor_critic.optionSelection.parameters(), args.lr, eps = args.eps)
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
    episode_rewards = torch.zeros([ 1])
    final_rewards = torch.zeros([1])
    optionSelection = 0
    options = torch.FloatTensor([-1] * num_options)
    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()
        options.cuda()
    
    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Choose Option 
            if options == -1:
                selection_value, options, option_log_prob, states = actor_critic.get_option(Variable(rollouts.observations[step], volatile=True),
                    Variable(rollouts.states[step], volatile=True),
                    Variable(rollouts.masks[step], volatile=True))
            print("option is:")
            print(options)

            # Sample actions
            value, action, action_log_prob, states = actor_critic.get_output(
                    options,
                    Variable(rollouts.observations[step], volatile=True),
                    Variable(rollouts.states[step], volatile=True),
                    Variable(rollouts.masks[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()
            print("action is:")
            print(action)

            # Termination 
            term_value, termination, termination_log_prob, _ = actor_critic.get_termination(
                options,
                Variable(rollouts.observations[step], volatile=True),
                    Variable(rollouts.states[step], volatile=True),
                    Variable(rollouts.masks[step], volatile=True))
            print("termination is:")
            print(termination)

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

            if termination:
                options = -1

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, reward, masks, options, termination)

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
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
                    options = Variable(rollouts.options[i])
                    adv_targ = Variable(advantages[i])
                    old_action_log_probs = rollouts.action_log_probs[i]
                    termination = Variable(rollouts.optionSelection[i])
                    # Use critic value of option nn to update option parameters
                    values, action_log_probs, dist_entropy, states = actor_critic.evaluate_option(
                        Variable(rollouts.observations[i]),
                        Variable(rollouts.states[i]),
                        Variable(rollouts.masks[i]),
                        Variable(rollouts.actions[i]), options)
                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)
                    value_loss = (Variable(self.returns[i]) - values).pow(2).mean()
                    optionOptimizer = optionOptimizers[options]
                    optionOptimizer.zero_grad()
                    (action_loss + value_loss).backward()
                    nn.utils.clip_grad_norm(optionOptimizers.parameters(), args.max_grad_norm)
                    optionOptimizers.step()

                    # Use critic value of option nn to obtain V_Omega. Not correct at the moment. Need to fix this part.
                    selection_log_prob = actor_critic.evaluate_selection(
                        Variable(rollouts.observations[i]),
                        Variable(rollouts.states[i]),
                        Variable(rollouts.masks[i]),
                        options)
                    V_Omega = selection_log_prob * values 

                    # Update termination parameters 
                    termination_log_prob = actor_critic.evaluate_termination(
                        Variable(rollouts.observations[i]),
                        Variable(rollouts.states[i]),
                        Variable(rollouts.masks[i]),
                        termination, 
                        options)
                    if termination:
                        termination_loss = - torch.exp(termination_log_prob) * V_Omega - (1 - torch.exp(termination_log_prob)) * values
                    else:
                        termination_loss = - (1 - torch.exp(termination_log_prob)) * V_Omega - torch.exp(termination_log_prob) * values
                    terminationOptimizer = terminationOptimizers[options]
                    terminationOptimizer.zero_grad()
                    (termination_loss).backward()
                    nn.utils.clip_grad_norm(terminationOptimizers.parameters(), args.max_grad_norm)
                    terminationOptimizer.step()

                    if i != 0 and rollouts.optionSelection[i - 1]:
                        optionSelectionOptimizer.zero_grad()
                        (V_Omega).backward()
                        nn.utils.clip_grad_norm(optionSelectionOptimizer.parameters(), args.max_grad_norm)
                        optionSelectionOptimizer.step()

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
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass

if __name__ == "__main__":
    main()
