import time
from collections import deque

import numpy as np
from stable_baselines import SAC
from stable_baselines import logger
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.common import TensorboardWriter

import scipy.stats
from statistics import mean
from utils.joystick import JoyStick

class SACWithVAE(SAC):
    """
    Custom version of Soft Actor-Critic (SAC) to use it with donkey car env.
    It is adapted from the stable-baselines version.

    Notable changes:
    - optimization is done after each episode and not at every step
    - this version is integrated with teleoperation

    """
    def _train_step(self, step, writer, learning_rate):
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }

        # out  = [policy_loss, qf1_loss, qf2_loss,
        #         value_loss, qf1, qf2, value_fn, logp_pi,
        #         self.entropy, policy_train_op, train_values_op]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + self.step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]

        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy


    def optimize(self, step, writer, current_lr):
        """
        Do several optimization steps to update the different networks.
        
        :param step: (int) current timestep
        :param writer: (TensorboardWriter object)
        :param current_lr: (float) Current learning rate
        :return: ([np.ndarray]) values used for monitoring
        """
        train_start = time.time()
        mb_infos_vals = []
        for grad_step in range(self.gradient_steps):
            if step < self.batch_size or step < self.learning_starts:
                break
            if len(self.replay_buffer) < self.batch_size:
                break

            self.n_updates += 1
            # Update policy and critics (q functions)
            mb_infos_vals.append(self._train_step(step, writer, current_lr))

            if (step + grad_step) % self.target_update_interval == 0:
                # Update target network
                self.sess.run(self.target_update_op)
        if self.n_updates > 0:
            # print("SAC training duration: {:.2f}s".format(time.time() - train_start))
            pass
        return mb_infos_vals
    
    def importance_sampling_ratio(self, proba_behavior_policy, proba_target_policy):
        EPS = 1e-10
        if proba_behavior_policy == 0:
            ratio = (proba_target_policy + EPS)/(proba_behavior_policy + EPS)
        else:
            ratio = proba_target_policy/proba_behavior_policy
        return ratio


    def learn_jirl(self, total_timesteps, joystick=None, callback=None, 
              seed=None, log_interval=1, tb_log_name="SAC", 
              print_freq=100, base_policy=None, stochastic_actor=True, 
              expert_guidance_steps=50000, save_path=None):

        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:
            # Add path to model in this function
            self._setup_learn(seed)

            # Joystick object
            js = JoyStick()	

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            
            episode_rewards = [0.0]
            
            # Reset the environment
            obs = self.env.reset()
            
            # Book keeping
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            ep_len = 0
            self.n_updates = 0
            n_crashes = 0
            infos_values = []
            mb_infos_vals = []
            pred_action_info = deque(maxlen=20)
            mean_info = deque(maxlen=50)
            std_info = deque(maxlen=50)
            throttle_info = deque(maxlen=1000)

            is_action_expert = False
            is_action_actor = True

            was_last_action_actor = False

            last_action_actor = None
            last_obs = None
            # steps in which expert takes control
            expert_control_steps = []
            state = {} # for the imitation learning agent

            MAX_LEN = 10
            
            is_ratios_target_expert = deque(maxlen=MAX_LEN) # IS ratios over the last few steps
            is_ratios_target_actor = deque(maxlen=MAX_LEN)

            EPS = 1e-10

            # Stats for plotting
            rew_per_step = []
            rew_per_step_rl = []
            rl_control = []

            # Buffer to control the threshold dynamically
            thresh_buffer = deque(maxlen=1000)
            std_buffer = deque(maxlen=10000)
            mean_buffer = deque(maxlen=10000)
        
            import time
            start_time = time.time()
            try:
                for step in range(total_timesteps):
                    # Compute current learning_rate
                    frac = 1.0 - step / total_timesteps
                    current_lr = self.learning_rate(frac)

                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break

                    # Get prediction from base policy
                    steerCmd = float(base_policy.predict(obs)[0][0])
    #                 print("Steering from IL: ", steerCmd)
                    throttleCmd = - 1
                    action_expert = [steerCmd, throttleCmd]
                    # mean_exp, std_exp = il_model.get_proba_actions(state)
                    # print(scipy.stats.multivariate_normal(mean = mean, cov = std).pdf(action_expert))

                    # Test with hard coded variance
                    # std_exp = [0.1, 0.1]
                    # proba_expert_policy = scipy.stats.norm(mean_exp[0], std_exp[0]).pdf(action_expert[0])
                    # proba_expert_policy = scipy.stats.norm(mean_exp[0], std_exp[0]).cdf(action_expert[0] + EPS) - scipy.stats.norm(mean_exp[0], std_exp[0]).cdf(action_expert[0] - EPS)
                    # if 2*np.pi*np.prod(std) <= 1:
                    #     proba_expert_policy = 2*np.pi*np.prod(std)*scipy.stats.multivariate_normal(mean = mean, cov = std).pdf(action_expert)
                    # else:
                    #     proba_expert_policy = scipy.stats.multivariate_normal(mean = mean, cov = std).pdf(action_expert)

                    ## ====== Test code snippet ======
                    # action_expert, _ = model.predict(obs, deterministic=True)
                    # new_obs, reward, done, info = self.env.step(action_expert)
                    ## ===============================

                    if not stochastic_actor:
                        action_actor = self.policy_tf.step(obs[None], deterministic=True).flatten()
                    else:
                        action_actor = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    
                    
                    if step >= expert_guidance_steps:
                        action_actor = self.policy_tf.step(obs[None], deterministic=True).flatten()

                    mean_act, std_act = self.policy_tf.proba_step(obs[None])
                    # print(scipy.stats.multivariate_normal(mean = mean.flatten(), cov = std.flatten()).pdf(action_actor))

                    proba_actor_policy = scipy.stats.norm(mean_act.flatten()[0], std_act.flatten()[0]).pdf(action_actor[0])

                    proba_expert_policy = scipy.stats.norm(mean_act.flatten()[0], std_act.flatten()[0]).pdf(action_expert[0])
                    # proba_actor_policy = scipy.stats.norm(mean_act.flatten()[0], std_act.flatten()[0]).cdf(action_actor[0] + EPS) - scipy.stats.norm(mean_act.flatten()[0], std_act.flatten()[0]).cdf(action_actor[0] - EPS)
                    # if 2*np.pi*np.prod(std) <= 1:
                    #     proba_actor_policy = 2*np.pi*np.prod(std.flatten())*scipy.stats.multivariate_normal(mean = mean.flatten(), cov = std.flatten()).pdf(action_actor)
                    # else:
                    #     proba_actor_policy = scipy.stats.multivariate_normal(mean = mean.flatten(), cov = std.flatten()).pdf(action_actor)
                    # Update entropy buffer
                    std_buffer.append(std_act)
                    # Update mean difference buffer
                    mean_buffer.append(np.linalg.norm(mean_act - action_expert))
                    # mean_buffer.append(np.linalg.norm(action_actor - action_expert))
                    rho = round(float(step)/expert_guidance_steps, 2)
                    # THRESH = (1 - rho) * (scipy.stats.norm(0, 0.1).pdf(0) - 1.0)**MAX_LEN
                    # _THRESH = (1 - rho) * (scipy.stats.norm(0, 0.1).pdf(0) - 2.0)

                    _THRESH = (np.mean(std_buffer) + np.mean(mean_buffer)) * (1 - rho)
                    THRESH = _THRESH**MAX_LEN

                    if step >= expert_guidance_steps:
                        # Only let the RL control the car
                        # If this doesn't work, tune MAX_LEN
                        THRESH = _THRESH = 0

                    if js.is_on():
                         ## =====================================
                         ## MANUAL CONTROL
                         ## =====================================
                         # Execute commands from the joystick in the environment
                         action_js = [js.get_steer(), -1]
                         new_obs, reward, done, info = self.env.step(action_js)

                         # Store transition in the replay buffer.
                         self.replay_buffer.add(obs, action_js, reward, new_obs, float(done))

                         ## ==========================================
                         sigma_p = 0.01
                         reward_hat = reward*np.exp(-np.linalg.norm(action_actor - action_js)/sigma_p)
                         self.replay_buffer.add(obs, action_actor, reward_hat, new_obs, float(done))
                         ## ==========================================

                         if was_last_action_actor:
                             # Train the actor when the expert's actions are executed
                             # mb_infos_vals = self.optimize(step, writer, current_lr)
                             penalty = -1 #-10
                             self.replay_buffer.add(last_obs, last_action_actor, penalty, obs, float(done))
                             is_ratios_target_expert = deque(maxlen=MAX_LEN)
                             was_last_action_actor = False
                             last_action_actor = None
                             last_obs = None

                         is_action_actor = False

                         # print("Actor IS ratio: ", is_ratio)

                         # if ep_len > 700:
                         #     print("Expert: ", np.prod(is_ratios_target_actor))
                         if (len(is_ratios_target_actor) == MAX_LEN) and np.all([(p > _THRESH) for p in is_ratios_target_actor]):
                             # Switch control to actor in the next step
                             is_action_actor = True

                         rew_per_step_rl.append(0.0)
                         rl_control.append(0)

    #                 else:
                    elif is_action_actor:
                        ## =====================================
                        ## RL CONTROL
                        ## =====================================
                        # Execute actor's actions in the environment
                        new_obs, reward, done, info = self.env.step(action_actor)

                        # Update IS ratiowill need to
                        is_ratio = self.importance_sampling_ratio(1.0, proba_expert_policy)
                        is_ratios_target_expert.append(is_ratio)

                        # Store transition in the replay buffer.
                        self.replay_buffer.add(obs, action_actor, reward, new_obs, float(done)) 

                        if not was_last_action_actor:
                            is_ratios_target_actor = deque(maxlen=MAX_LEN)

                        is_action_actor = True

                        # print("Actor: ", np.prod(is_ratios_target_expert))
                        # Per step safety check
                        if is_ratio < _THRESH:
                            # Switch control to the expert
                            is_action_actor = False

                        # Safe ty check for a sequence of states
                        if (len(is_ratios_target_actor) == MAX_LEN) and np.all([(p > _THRESH) for p in is_ratios_target_actor]):
                        #if (len(is_ratios_target_expert) == MAX_LEN) and (np.prod(is_ratios_target_expert) <= THRESH):
                            # Switch control to expert in the next step
                            is_action_actor = False

                        was_last_action_actor = True
                        last_action_actor = action_actor
                        last_obs = obs

                        rew_per_step_rl.append(reward)
                        rl_control.append(1)

                    else:
                        ## =======================================
                        ## EXPERT CONTROL
                        ## =======================================
                        # Execute expert action in the environment
                        new_obs, reward, done, info = self.env.step(action_expert)
                        # Update IS ratio
                        # is_ratio = self.importance_sampling_ratio(1.0, proba_actor_policy)
                        is_ratio = self.importance_sampling_ratio(1.0, proba_expert_policy)
                        is_ratios_target_actor.append(is_ratio)

                        # print("Expert ", is_ratio)

                        # Store transition in the replay buffer.
                        self.replay_buffer.add(obs, action_expert, reward, new_obs, float(done))

                        ## ==========================================
                        # # NOTE: Figure out what's going wrong here
                        # # Without the penalized reward the policy diverges (mean doesn't go towards 0
                        # # Also test with stochastic actions from the RL policy
                        # # # Add penalized reward to actor's action
                        # # r_hat: penalized reward
                        sigma_p = 0.01
                        reward_hat = reward*np.exp(-np.linalg.norm(action_actor - action_expert)/sigma_p)
                        self.replay_buffer.add(obs, action_actor, reward_hat, new_obs, float(done))
                        ## ==========================================

                        if was_last_action_actor:
                            # Train the actor when the expert's actions are executed
                            # mb_infos_vals = self.optimize(step, writer, current_lr)
                            penalty = -1 #-10
                            self.replay_buffer.add(last_obs, last_action_actor, penalty, obs, float(done))
                            is_ratios_target_expert = deque(maxlen=MAX_LEN)
                            was_last_action_actor = False
                            last_action_actor = None
                            last_obs = None

                        is_action_actor = False

                        # print("Actor IS ratio: ", is_ratio)

                        # if ep_len > 700:
                        #     print("Expert: ", np.prod(is_ratios_target_actor))

#                         if (len(is_ratios_target_actor) == MAX_LEN) and (np.prod(is_ratios_target_actor) > THRESH):
                        if (len(is_ratios_target_actor) == MAX_LEN) and np.all([(p > _THRESH) for p in is_ratios_target_actor]):
                            # Switch control to actor in the next step
                            is_action_actor = True

                        rew_per_step_rl.append(0.0)
                        rl_control.append(0)
                
                    throttle_info.append(float(self.env.last_throttle))
                    rew_per_step.append(reward)

                    pred_action_info.append(np.abs(action_actor[0] - action_expert[0]))
                    # mean_info.append([mean_exp[0], mean_act.flatten()[0]])
                    # std_info.append([std_exp[0], std_act.flatten()[0]])

                    ep_len += 1
                    obs = new_obs

                    if ep_len % 400 == 0:
                        print("Mean error pred actions: {}".format(np.mean(pred_action_info)))
                        print("Mean difference: {}".format(np.mean(mean_buffer)))
                        print("Mean std: {}".format(np.mean(std_buffer)))
                        # print("Mean: ", [np.mean([x[0] for x in mean_info]), np.mean([x[1] for x in mean_info])])
                        # print("Std: ", [np.mean([x[0] for x in std_info]), np.mean([x[1] for x in std_info])])
                        # print(np.prod(is_ratios_target_actor))

                    # Train every step  ---under consideratioon
                    if (ep_len % 400) == 0:
                        self.env.jet.apply_throttle(0)
                        mb_infos_vals = self.optimize(step, writer, current_lr)

    #                 if print_freq > 0 and ep_len % print_freq == 0 and ep_len > 0:
    #                     print("{} steps".format(ep_len))

                    # Retrieve reward and episode length if using Monitor wrapper
                    maybe_ep_info = info.get('episode')
                    if maybe_ep_info is not None:
                        ep_info_buf.extend([maybe_ep_info])

                    if writer is not None:
                        # Write reward per episode to tensorboard
                        ep_reward = np.array([reward]).reshape((1, -1))
                        ep_done = np.array([done]).reshape((1, -1))
                        self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                              ep_done, writer, step)

                    episode_rewards[-1] += reward

                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                    if len(episode_rewards[-101:-1]) == 0:
                        mean_reward = -np.inf
                    else:
                        mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                    if len(rl_control) < 1000:
                        mean_rl_control = round(100 * float(np.mean(rl_control)), 3)
                    else:
                        mean_rl_control = round(100 * float(np.mean(rl_control[-1001:-1])), 3)

                    num_episodes = len(episode_rewards)


                    if self.verbose >= 1 and (ep_len % 400) == 0:
                        logger.logkv("episodes", num_episodes)
                        logger.logkv("mean 100 episode reward", mean_reward)
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                        logger.logkv("n_updates", self.n_updates)
                        logger.logkv("current_lr", current_lr)
                        logger.logkv("mean RL control percent", mean_rl_control)
                        logger.logkv("mean of throttle values", mean(throttle_info))
                        logger.logkv("time elapsed", int(time.time() - start_time))
                        #logger.logkv("n_crashes", n_crashes)
                        if len(infos_values) > 0:
                            for (name, val) in zip(self.infos_names, infos_values):
                                logger.logkv(name, val)
                        logger.logkv("total timesteps", step)
                        logger.dumpkvs()
                        # Reset infos:
                        infos_values = []
            except KeyboardInterrupt:
                print("Exiting")
                self.env.reset()
                import sys
                sys.exit(0)
                    
            # Use last batch
            print("Final optimization before saving")
            self.env.reset()
            mb_infos_vals = self.optimize(step, writer, current_lr)

            # save stats
            np.save(save_path + '/episode_reward', episode_rewards)
            np.save(save_path + '/stepwise_reward', rew_per_step)
            np.save(save_path + '/stepwise_reward_rl', rew_per_step_rl)
            print("Saving complete. Give a keyboard interrupt to end")
        return self


    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=10, tb_log_name="SAC", reset_num_timesteps=True,
              prioritized_replay=False, stochastic_actor=False, save_path=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) as writer:

            self._setup_learn(seed)

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            episode_rewards = [0.0]
            obs = self.env.reset()
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []

            # Stats for plotting
            rew_per_step = []

            for step in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy.
                if self.num_timesteps < self.learning_starts:
                    action = self.env.action_space.sample()
                    # No need to rescale when sampling random action
                    rescaled_action = action
                else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done))
                obs = new_obs
                rew_per_step.append(reward)

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)

                if step % self.train_freq == 0:
                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        if self.num_timesteps < self.batch_size or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(step, writer, current_lr))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                episode_rewards[-1] += reward
                if done:
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                self.num_timesteps += 1
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                    logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []

            # Save book keeping stats
            np.save(save_path + '/episode_reward', episode_rewards)    
            np.save(save_path + '/stepwise_reward', rew_per_step)
            
            return self
