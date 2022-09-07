# Some parts adapted from https://github.com/alexis-jacq/LOLA_DiCE/blob/master/ipd_DiCE.py
# Some parts adapted from Chris MOFOS repo

# import jnp
import math
# import jnp.nn as nn
# from jnp.distributions import Categorical
import numpy as np
import argparse
import os
import datetime

import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
import functools
import optax
from functools import partial

import flax
from flax import linen as nn
import jax.numpy as jnp
from typing import NamedTuple, Callable, Any
from flax.training.train_state import TrainState

from flax.training import checkpoints

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


from coin_game_jax import CoinGame
from ipd_jax import IPD

def reverse_cumsum(x, axis):
    return x + jnp.sum(x, axis=axis, keepdims=True) - jnp.cumsum(x, axis=axis)


device = 'cpu'

# DiCE operator
@jit
def magic_box(x):
    return jnp.exp(x - jax.lax.stop_gradient(x))



@jit
def update_gae_with_delta_backwards(gae, delta):
    gae = gae * args.gamma * args.gae_lambda + delta
    return gae, gae


@jit
def get_gae_advantages(rewards, values, next_val_history):
    deltas = rewards + args.gamma * jax.lax.stop_gradient(
        next_val_history) - jax.lax.stop_gradient(values)

    # print(rewards)
    # print(rewards.shape)
    #
    # print(next_val_history.shape)
    # print(values.shape)
    # print(deltas)
    # print(deltas.shape)

    gae = jnp.zeros_like(deltas[0, :])

    # print(gae.shape)
    # print(gae)

    deltas = jnp.flip(deltas, axis=0)
    gae, flipped_advantages = jax.lax.scan(update_gae_with_delta_backwards, gae, deltas, deltas.shape[0])
    advantages = jnp.flip(flipped_advantages, axis=0)

    # print(gae.shape)
    # print(advantages.shape)


    # TODO REWRITE THIS LOOP AS LAX.SCAN AND THEN TEST THAT IT COMPILES REASONABLY WITH VALUE FUNC.
    # for i in range(deltas.shape[0] - 1, -1, -1):
    #     gae = gae * args.gamma * args.gae_lambda + deltas[i, :]
    #     advantages = advantages.at[i, :].set(gae)

    # print(advantages)
    return advantages



@jit
def dice_objective(self_logprobs, other_logprobs, rewards, values, end_state_v):
    # print(self_logprobs)
    # self_logprobs = jnp.stack(self_logprobs, dim=1)
    # other_logprobs = jnp.stack(other_logprobs, dim=1)

    # rewards = jnp.stack(rewards, dim=1)

    # print(rewards)
    # print(rewards.shape)
    # print(rewards.size)

    # if args.env == 'coin':
    #     rewards = rewards.squeeze(-1)

    # print(self_logprobs.shape)
    # print(other_logprobs.shape)
    # print(rewards.shape)

    # apply discount:
    cum_discount = jnp.cumprod(args.gamma * jnp.ones(rewards.shape),
                                 axis=0) / args.gamma
    discounted_rewards = rewards * cum_discount


    # print(cum_discount)
    # print(discounted_rewards)

    # stochastics nodes involved in rewards dependencies:
    dependencies = jnp.cumsum(self_logprobs + other_logprobs, axis=0)

    # logprob of all stochastic nodes:
    stochastic_nodes = self_logprobs + other_logprobs

    use_loaded_dice = False
    if use_baseline:
        use_loaded_dice = True

    if use_loaded_dice:
        next_val_history = jnp.zeros((args.rollout_len, args.batch_size))

        next_val_history = next_val_history.at[:args.rollout_len - 1, :].set(values[1:args.rollout_len, :])
        next_val_history = next_val_history.at[-1, :].set(end_state_v)

        # print(next_val_history.shape)
        # print(next_val_history[:args.rollout_len - 1, :].shape)
        # print(values[1:args.rollout_len, :].shape)
        # print(next_val_history[-1, :].shape)
        # print(end_state_v.shape)

        if args.zero_vals:
            next_val_history = jnp.zeros_like(next_val_history)
            values = jnp.zeros_like(values)

        advantages = get_gae_advantages(rewards, values, next_val_history)

        # print(advantages.shape)

        discounted_advantages = advantages * cum_discount

        # print(cum_discount.shape)

        deps_up_to_t = (jnp.cumsum(stochastic_nodes, axis=0))

        # print(deps_up_to_t.shape)

        deps_less_than_t = deps_up_to_t - stochastic_nodes  # take out the dependency in the given time step

        # print(stochastic_nodes.shape)
        # print(deps_less_than_t.shape)

        # Look at Loaded DiCE and GAE papers to see where this formulation comes from
        loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(
            deps_less_than_t)) * discounted_advantages).sum(axis=0).mean()

        # print(((magic_box(deps_up_to_t) - magic_box(
            # deps_less_than_t)) * discounted_advantages).shape)

        dice_obj = loaded_dice_rewards

    else:
        # dice objective:
        # REMEMBER that in this jax code the axis 0 is the rollout_len (number of time steps in the environment)
        # and axis 1 is the batch.
        dice_obj = jnp.mean(
            jnp.sum(magic_box(dependencies) * discounted_rewards, axis=0))


    return -dice_obj  # want to minimize -objective


@jit
def dice_objective_plus_value_loss(self_logprobs, other_logprobs, rewards, values, end_state_v):
    # Essentially a wrapper function for the objective to put all the control flow in one spot
    # The reasoning behind this function here is that the reward_loss has a stop_gradient
    # on all of the nodes related to the value function
    # and the value function has no nodes related to the policy
    # Then we can actually take the respective grads like the way I have things set up now
    # And I should be able to update both policy and value functions

    reward_loss = dice_objective(self_logprobs, other_logprobs, rewards, values, end_state_v)

    if use_baseline:
        val_loss = value_loss(rewards, values, end_state_v)
        return reward_loss + val_loss
    else:
        return reward_loss


@jit
def value_loss(rewards, values, final_state_vals):

    final_state_vals = jax.lax.stop_gradient(final_state_vals)

    # if args.env == 'coin':
    #     rewards = rewards.squeeze(-1)


    discounts = jnp.cumprod(args.gamma * jnp.ones(rewards.shape),
                                 axis=0) / args.gamma

    # print(rewards)

    gamma_t_r_ts = rewards * discounts

    G_ts = reverse_cumsum(gamma_t_r_ts, axis=0)
    R_ts = G_ts / discounts
    # print(G_ts) # TODO check the actual numbers, check that the cumsum is correct. Get rid of jits and whatever, just try something outside of a lax.scan loop first

    # print(discounts)
    # print(args.gamma * jnp.flip(discounts, axis=0))
    # print(final_state_vals)

    final_val_discounted_to_curr = (args.gamma * jnp.flip(discounts, axis=0)) * final_state_vals
    # print(final_val_discounted_to_curr)
    # print(R_ts)



    # You DO need a detach on these. Because it's the target - it should be detached. It's a target value.
    # Essentially a Monte Carlo style type return for R_t, except for the final state we also use the estimated final state value.
    # This becomes our target for the value function loss. So it's kind of a mix of Monte Carlo and bootstrap, but anyway you need the final value
    # because otherwise your value calculations will be inconsistent
    values_loss = (R_ts + final_val_discounted_to_curr - values) ** 2


    values_loss = values_loss.sum(axis=0).mean()

    # print("Values loss")
    # print(values_loss)
    return values_loss


@jit
def act_w_iter_over_obs(stuff, env_batch_obs):
    key, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, h_p, h_v = stuff
    key, subkey = jax.random.split(key)
    act_args = (subkey, env_batch_obs, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, h_p, h_v)
    act_args, act_aux = act(act_args, None)
    _, env_batch_obs, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, h_p, h_v = act_args
    stuff = (key, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, h_p, h_v)
    return stuff, act_aux

@jit
def act(stuff, unused ):
    key, env_batch_states, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, h_p, h_v = stuff
    # print(env_batch_states)

    h_p, logits = th_p_trainstate.apply_fn(th_p_trainstate_params, env_batch_states, h_p)

    # print(h_p)
    # print(logits)

    categorical_act_probs = jax.nn.softmax(logits)
    if use_baseline:
        h_v, values = th_v_trainstate.apply_fn(th_v_trainstate_params, env_batch_states, h_v)
        ret_vals = values.squeeze(-1)
    else:
        h_v, values = None, None
        ret_vals = None

    # actions = jax.random.categorical(key, logits)
    # dist = Categorical(categorical_act_probs)
    # actions = dist.sample()
    # TODO how does RNG work with tfd?
    dist = tfd.Categorical(logits=logits)
    key, subkey = jax.random.split(key)
    actions = dist.sample(seed=subkey)

    log_probs_actions = dist.log_prob(actions)

    # print(logits.shape)
    # print(actions)
    # print(log_probs_actions)

    stuff = (key, env_batch_states, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, h_p, h_v)
    aux = (actions, log_probs_actions, ret_vals, h_p, h_v, categorical_act_probs, logits)

    return stuff, aux

    # if ret_logits:
    #     return actions, log_probs_actions, ret_vals, h_p, h_v, categorical_act_probs, logits
    # return actions, log_probs_actions, ret_vals, h_p, h_v, categorical_act_probs



class RNN(nn.Module):
    num_outputs: int
    num_hidden_units: int

    def setup(self):
        if args.layers_before_gru >= 1:
            self.linear1 = nn.Dense(features=self.num_hidden_units)
        if args.layers_before_gru >= 2:
            self.linear2 = nn.Dense(features=self.num_hidden_units)
        self.GRUCell = nn.GRUCell()
        self.linear_end = nn.Dense(features=self.num_outputs)

    def __call__(self, x, carry):
        if args.layers_before_gru >= 1:
            x = self.linear1(x)
            x = nn.relu(x)
        if args.layers_before_gru >= 2:
            x = self.linear2(x)

        carry, x = self.GRUCell(carry, x)
        outputs = self.linear_end(x)
        return carry, outputs

    # def initialize_carry(self):
    #     return self.GRUCell.initialize_carry(
    #         jax.random.PRNGKey(0), (), self.num_hidden_units
    #     )


@jit
def get_policies_for_states(key, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, obs_hist):

    h_p = jnp.zeros((args.batch_size, args.hidden_size))
    h_v = None
    if use_baseline:
        h_v = jnp.zeros((args.batch_size, args.hidden_size))

    key, subkey = jax.random.split(key)

    act_args = (subkey, th_p_trainstate, th_p_trainstate_params,
                th_v_trainstate, th_v_trainstate_params, h_p, h_v)
    # Note that I am scanning using xs = obs_hist. Then the scan should work through the
    # array of obs.
    obs_hist_for_scan = jnp.stack(obs_hist[:args.rollout_len], axis=0)

    act_args, aux_lists = jax.lax.scan(act_w_iter_over_obs, act_args, obs_hist_for_scan, args.rollout_len)
    # act_args, aux_lists = jax.lax.scan(act_w_iter_over_obs, act_args, obs_hist_for_scan, obs_hist_for_scan.shape[0])

    a_list, lp_list, v_list, h_p_list, h_v_list, cat_act_probs_list, logits_list = aux_lists


    return cat_act_probs_list


@jit
def get_policies_for_states_onebatch(key, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, obs_hist):

    h_p = jnp.zeros((1, args.hidden_size))
    h_v = None
    if use_baseline:
        h_v = jnp.zeros((1, args.hidden_size))

    key, subkey = jax.random.split(key)

    act_args = (subkey, th_p_trainstate, th_p_trainstate_params,
                th_v_trainstate, th_v_trainstate_params, h_p, h_v)
    # Note that I am scanning using xs = obs_hist. Then the scan should work through the
    # array of obs.
    obs_hist_for_scan = jnp.stack(obs_hist[:len(obs_hist)], axis=0)

    # act_args, aux_lists = jax.lax.scan(act_w_iter_over_obs, act_args, obs_hist_for_scan, args.rollout_len)
    act_args, aux_lists = jax.lax.scan(act_w_iter_over_obs, act_args, obs_hist_for_scan, obs_hist_for_scan.shape[0])

    a_list, lp_list, v_list, h_p_list, h_v_list, cat_act_probs_list, logits_list = aux_lists


    return cat_act_probs_list


@jit
# def env_step_from_actions(stuff, unused):
#     # TODO I may have to pass in one agent params anyway and kind of follow the env_step pattern
#     # I may need a other_agent=2 switch again...
#     # Plan out the whole thing before coding
#     # TODO anyway all this needs to go into the eval_progress and eval_vs_fixed... functions
#     skenv, env_state, a1, a2 = stuff
#     env_state, new_obs, (r1, r2), (rr_match, rb_match, br_match, bb_match) = vec_env_step(env_state, a1, a2, skenv)
#     stuff = (skenv, env_state, a1, a2)
#     aux = (r1, r2, rr_match, rb_match, br_match, bb_match)
#     return stuff, aux

@jit
def env_step(stuff, unused):
    # TODO should make this agent agnostic? Or have a flip switch? Can reorganize later
    key, env_state, obs1, obs2, \
    trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params, \
    trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params, \
    h_p1, h_v1, h_p2, h_v2 = stuff
    key, sk1, sk2, skenv = jax.random.split(key, 4)
    act_args1 = (sk1, obs1, trainstate_th1, trainstate_th1_params,
                trainstate_val1, trainstate_val1_params, h_p1, h_v1)
    act_args2 = (sk2, obs2, trainstate_th2, trainstate_th2_params,
                 trainstate_val2, trainstate_val2_params, h_p2, h_v2)
    stuff1, aux1 = act(act_args1, None)
    a1, lp1, v1, h_p1, h_v1, cat_act_probs1, logits1 = aux1
    stuff2, aux2 = act(act_args2, None)
    a2, lp2, v2, h_p2, h_v2, cat_act_probs2, logits2 = aux2
    # a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(act_args1, None)
    # a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(sk2, obs2,
    #                                               trainstate_th2,
    #                                               trainstate_th2_params,
    #                                               trainstate_val2,
    #                                               trainstate_val2_params,
    #                                               h_p2,
    #                                               h_v2)
    skenv = jax.random.split(skenv, args.batch_size)

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, skenv)
    # print(new_obs)
    # env_state, new_obs, (r1, r2) = env.step(env_state, a1, a2, skenv)
    obs1 = new_obs
    obs2 = new_obs
    # other_memory.add(lp2, lp1, v2, r2)


    stuff = (key, env_state, obs1, obs2,
             trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
             trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,
             h_p1, h_v1, h_p2, h_v2)
    # aux = (cat_act_probs2, obs2, self_lp, other_lp, v2, r2)

    aux1 = (cat_act_probs1, obs1, lp1, lp2, v1, r1, a1, a2)

    aux2 = (cat_act_probs2, obs2, lp2, lp1, v2, r2, a2, a1)

    return stuff, (aux1, aux2, aux_info)

@partial(jit, static_argnums=(9))
def do_env_rollout(key, trainstate_th1, trainstate_th1_params, trainstate_val1,
             trainstate_val1_params,
             trainstate_th2, trainstate_th2_params, trainstate_val2,
             trainstate_val2_params, agent_for_state_history):
    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    # env_state, obsv = env.reset(subkey)
    env_state, obsv = vec_env_reset(env_subkeys)

    obs1 = obsv
    obs2 = obsv

    # print(obsv.shape)

    # other_memory = Memory()
    h_p1, h_p2, h_v1, h_v2 = get_init_hidden_states()

    # inner_agent_cat_act_probs = []
    # TODO something like this for the vals?
    # state_history_for_vals = []
    # state_history_for_vals.append(obs1)
    unfinished_state_history = []
    if agent_for_state_history == 2:
        unfinished_state_history.append(obs2)
    else:
        assert agent_for_state_history == 1
        unfinished_state_history.append(obs1)

    stuff = (key, env_state, obs1, obs2,
             trainstate_th1, trainstate_th1_params, trainstate_val1,
             trainstate_val1_params,
             trainstate_th2, trainstate_th2_params, trainstate_val2,
             trainstate_val2_params,
             h_p1, h_v1, h_p2, h_v2)

    stuff, aux = jax.lax.scan(env_step, stuff, None, args.rollout_len)

    return stuff, aux, unfinished_state_history

@partial(jit, static_argnums=(11))
def in_lookahead(key, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
                 trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,
                 old_trainstate_th, old_trainstate_val,
                 other_agent=2, inner_agent_pol_probs_old=None, inner_agent_state_history_ref=None):

    # keys = jax.random.split(key, args.batch_size + 1)
    # key, env_subkeys = keys[0], keys[1:]
    #
    # # env_state, obsv = env.reset(subkey)
    # env_state, obsv = vec_env_reset(env_subkeys)
    #
    # obs1 = obsv
    # obs2 = obsv
    #
    # # print(obsv.shape)
    #
    # # other_memory = Memory()
    # h_p1, h_p2 = (
    #     jnp.zeros((args.batch_size, args.hidden_size)),
    #     jnp.zeros((args.batch_size, args.hidden_size))
    # )
    #
    # h_v1, h_v2 = None, None
    # if use_baseline:
    #     h_v1, h_v2 = (
    #         jnp.zeros((args.batch_size, args.hidden_size)),
    #         jnp.zeros((args.batch_size, args.hidden_size))
    #     )
    #
    # # inner_agent_cat_act_probs = []
    # inner_agent_state_history = []
    # if other_agent == 2:
    #     inner_agent_state_history.append(obs2)
    # else:
    #     inner_agent_state_history.append(obs1)
    #
    #
    # stuff = (key, env_state, obs1, obs2,
    #          trainstate_th1, trainstate_th1_params, trainstate_val1,
    #          trainstate_val1_params,
    #          trainstate_th2, trainstate_th2_params, trainstate_val2,
    #          trainstate_val2_params,
    #          h_p1, h_v1, h_p2, h_v2)
    #
    # stuff, aux = jax.lax.scan(env_step, stuff, None, args.rollout_len)

    stuff, aux, unfinished_inner_agent_state_history = do_env_rollout(key, trainstate_th1, trainstate_th1_params, trainstate_val1,
             trainstate_val1_params,
             trainstate_th2, trainstate_th2_params, trainstate_val2,
             trainstate_val2_params, agent_for_state_history=other_agent)
    aux1, aux2, aux_info = aux

    inner_agent_state_history = unfinished_inner_agent_state_history

    key, env_state, obs1, obs2, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,\
    trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params, h_p1, h_v1, h_p2, h_v2 = stuff

    key, subkey1, subkey2 = jax.random.split(key, 3)

    # TODO remove the redundancies
    if other_agent == 2:
        cat_act_probs2_list, obs2_list, lp2_list, lp1_list, v2_list, r2_list, a2_list, a1_list = aux2

        # inner_agent_cat_act_probs.extend(cat_act_probs2_list)
        inner_agent_state_history.extend(obs2_list)

        # act just to get the final state values
        act_args2 = (subkey2, obs2, trainstate_th2, trainstate_th2_params,
                     trainstate_val2, trainstate_val2_params, h_p2, h_v2)
        stuff2, aux2 = act(act_args2, None)
        a2, lp2, v2, h_p2, h_v2, cat_act_probs2, logits2 = aux2

        # a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(subkey2, obs2,
        #                                               trainstate_th2,
        #                                               trainstate_th2_params,
        #                                               trainstate_val2,
        #                                               trainstate_val2_params,
        #                                               h_p2, h_v2)
        end_state_v2 = v2

        inner_agent_objective = dice_objective_plus_value_loss(self_logprobs=lp2_list,
                                               other_logprobs=lp1_list,
                                               rewards=r2_list,
                                               values=v2_list,
                                               end_state_v=end_state_v2)

        # print(f"Inner Agent (Agent 2) episode return avg {r2_list.sum(axis=0).mean()}")


    else:
        assert other_agent == 1
        cat_act_probs1_list, obs1_list, lp1_list, lp2_list, v1_list, r1_list, a1_list, a2_list = aux1
        # inner_agent_cat_act_probs.extend(cat_act_probs1_list)
        inner_agent_state_history.extend(obs1_list)

        act_args1 = (subkey1, obs1, trainstate_th1, trainstate_th1_params,
                     trainstate_val1, trainstate_val1_params, h_p1, h_v1)
        stuff1, aux1 = act(act_args1, None)
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1, logits1 = aux1
        # a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(subkey1, obs1,
        #                                               trainstate_th1,
        #                                               trainstate_th1_params,
        #                                               trainstate_val1,
        #                                               trainstate_val1_params,
        #                                               h_p1, h_v1)
        end_state_v1 = v1

        inner_agent_objective = dice_objective_plus_value_loss(self_logprobs=lp1_list,
                                               other_logprobs=lp2_list,
                                               rewards=r1_list,
                                               values=v1_list,
                                               end_state_v=end_state_v1)

        # print(f"Inner Agent (Agent 1) episode return avg {r1_list.sum(axis=0).mean()}")

    key, sk1, sk2 = jax.random.split(key, 3)

    if args.old_kl_div:
        assert inner_agent_pol_probs_old is not None # TODO save this somewhere, use the batch of states and pass it through once
        assert inner_agent_state_history_ref is not None
        if other_agent == 2:
            inner_agent_pol_probs = get_policies_for_states(sk1,
                                                            trainstate_th2,
                                                            trainstate_th2_params,
                                                            trainstate_val2,
                                                            trainstate_val2_params,
                                                            inner_agent_state_history_ref)
            # We don't need gradient on the old one, so we can just use the trainstate.params
        else:
            inner_agent_pol_probs = get_policies_for_states(sk1,
                                                            trainstate_th1,
                                                            trainstate_th1_params,
                                                            trainstate_val1,
                                                            trainstate_val1_params,
                                                            inner_agent_state_history_ref)
    else:
        if other_agent == 2:
            inner_agent_pol_probs = get_policies_for_states(sk1,
                                                            trainstate_th2,
                                                            trainstate_th2_params,
                                                            trainstate_val2,
                                                            trainstate_val2_params,
                                                            inner_agent_state_history)
            # We don't need gradient on the old one, so we can just use the trainstate.params
        else:
            inner_agent_pol_probs = get_policies_for_states(sk1,
                                                            trainstate_th1,
                                                            trainstate_th1_params,
                                                            trainstate_val1,
                                                            trainstate_val1_params,
                                                            inner_agent_state_history)
        inner_agent_pol_probs_old = get_policies_for_states(sk2,
                                                            old_trainstate_th,
                                                            old_trainstate_th.params,
                                                            old_trainstate_val,
                                                            old_trainstate_val.params,
                                                            inner_agent_state_history)

    # Note that Kl Div right now is based on the state history of this episode
    # Passed through the policies of the current agent policy params and the old params
    # So what this means is that on each inner step, you get a fresh batch of data
    # For the KL Div calculation too
    # This I think should be more stable than before
    # This means you aren't limited to KL Div only on the 4000 or whatever batch
    # you got from the very beginning
    # And so you should get coverage on a wider range of the state space
    # in the same way that your updates are based on new rollouts too
    # If we do repeat train, then the repeat train KL Div should be based on the
    # initial trajectory
    # and then I have to figure out how to save the initial trajectory and reuse it in Jax.

    kl_div = kl_div_jax(inner_agent_pol_probs, inner_agent_pol_probs_old)
    # print(f"KL Div: {kl_div}")

    return inner_agent_objective + args.inner_beta * kl_div  # we want to min kl div


@jit
def kl_div_jax(curr, target):
    kl_div = (curr * (jnp.log(curr) - jnp.log(target))).sum(axis=-1).mean()
    return kl_div



@jit
def inner_step_get_grad_otheragent2(stuff, unused):
    key, trainstate_th1_, trainstate_th1_params, trainstate_val1_, trainstate_val1_params, \
    trainstate_th2_, trainstate_th2_params, trainstate_val2_, trainstate_val2_params, old_trainstate_th, old_trainstate_val, \
    inner_agent_pol_probs_old, inner_agent_state_history_ref = stuff
    key, subkey = jax.random.split(key)

    other_agent_obj_grad_fn = jax.grad(in_lookahead, argnums=[6, 8])

    grad_th, grad_v = other_agent_obj_grad_fn(subkey,
                                              trainstate_th1_,
                                              trainstate_th1_params,
                                              trainstate_val1_,
                                              trainstate_val1_params,
                                              trainstate_th2_,
                                              trainstate_th2_params,
                                              trainstate_val2_,
                                              trainstate_val2_params,
                                              old_trainstate_th,
                                              old_trainstate_val,
                                              other_agent=2,
                                              inner_agent_pol_probs_old=inner_agent_pol_probs_old,
                                              inner_agent_state_history_ref=inner_agent_state_history_ref)

    # update other's theta: NOTE HERE THIS IS JUST AN SGD UPDATE
    trainstate_th2_ = trainstate_th2_.apply_gradients(grads=grad_th)

    # TODO when value update the inner model? Do it at all? In old code I didn't update on inner loop but also I only used 1 inner step in most experiments
    if use_baseline:
        # Now this should be correct because I am using dice_objective_plus_value_loss
        # which has both the policy and the value loss together
        trainstate_val2_ = trainstate_val2_.apply_gradients(grads=grad_v)

    # Since we only need the final trainstate, and not every trainstate every step of the way, no need for aux here
    # Note the dot here (on agent 2) because we want to return the updated params
    stuff = (key, trainstate_th1_, trainstate_th1_params, trainstate_val1_, trainstate_val1_params,
             trainstate_th2_, trainstate_th2.params, trainstate_val2_, trainstate_val2.params,
             old_trainstate_th, old_trainstate_val, inner_agent_pol_probs_old, inner_agent_state_history_ref)
    aux = None

    return stuff, aux

@jit
def inner_step_get_grad_otheragent1(stuff, unused):
    key, trainstate_th1_, trainstate_th1_params, trainstate_val1_, trainstate_val1_params, \
    trainstate_th2_, trainstate_th2_params, trainstate_val2_, trainstate_val2_params, old_trainstate_th, old_trainstate_val, \
    inner_agent_pol_probs_old, inner_agent_state_history_ref  = stuff
    key, subkey = jax.random.split(key)

    other_agent_obj_grad_fn = jax.grad(in_lookahead,
                                       argnums=[2, 4])

    grad_th, grad_v = other_agent_obj_grad_fn(subkey,
                                              trainstate_th1_,
                                              trainstate_th1_params,
                                              trainstate_val1_,
                                              trainstate_val1_params,
                                              trainstate_th2_,
                                              trainstate_th2_params,
                                              trainstate_val2_,
                                              trainstate_val2_params,
                                              old_trainstate_th, old_trainstate_val,
                                              other_agent=1,
                                              inner_agent_pol_probs_old=inner_agent_pol_probs_old,
                                              inner_agent_state_history_ref=inner_agent_state_history_ref)

    # update other's theta: NOTE HERE THIS IS JUST AN SGD UPDATE

    trainstate_th1_ = trainstate_th1_.apply_gradients(grads=grad_th)

    # TODO when value update the inner model? Do it at all? In old code I didn't update on inner loop but also I only used 1 inner step in most experiments
    if use_baseline:
        # Now this should be correct because I am using dice_objective_plus_value_loss
        # which has both the policy and the value loss together
        trainstate_val1_ = trainstate_val1_.apply_gradients(grads=grad_v)

    # Since we only need the final trainstate, and not every trainstate every step of the way, no need for aux here
    # Note the dot here (on agent 1) because we want to return the updated params
    stuff = (key, trainstate_th1_, trainstate_th1_.params, trainstate_val1_, trainstate_val1_.params,
             trainstate_th2_, trainstate_th2_params, trainstate_val2_, trainstate_val2_params,
             old_trainstate_th, old_trainstate_val, inner_agent_pol_probs_old, inner_agent_state_history_ref)
    aux = None

    return stuff, aux


@jit
def inner_steps_plus_update_otheragent2(key, trainstate_th1, trainstate_th1_params,
                            trainstate_val1, trainstate_val1_params,
                            trainstate_th2, trainstate_th2_params,
                            trainstate_val2, trainstate_val2_params,
                            other_old_trainstate_th, other_old_trainstate_val):

    # TODO reinstate the use of the th1params everywhere
    # and then do this for the other agent perspective too
    # Finally put it all together.
    # in inner_step_get_grad_otheragent2...

    trainstate_th2_ = TrainState.create(apply_fn=trainstate_th2.apply_fn,
                                        params=trainstate_th2_params,
                                        tx=optax.sgd(
                                            learning_rate=args.lr_in))
    trainstate_val2_ = TrainState.create(apply_fn=trainstate_val2.apply_fn,
                                         params=trainstate_val2_params,
                                         tx=optax.sgd(
                                             learning_rate=args.lr_v))

    key, reused_subkey = jax.random.split(key)
    # reuse the subkey to get consistent trajectories for the first batch
    # This is only needed so I can be consistent with my previous pytorch code
    # And does not really have a theoretical or logical grounding really

    other_pol_probs_ref = None
    other_state_history_ref = None

    key, subkey = jax.random.split(key)

    if args.old_kl_div:
        stuff, aux, unfinished_state_history = do_env_rollout(reused_subkey,
                                                              trainstate_th1,
                                                              trainstate_th1_params,
                                                              trainstate_val1,
                                                              trainstate_val1_params,
                                                              trainstate_th2_,
                                                              trainstate_th2_.params,
                                                              trainstate_val2_,
                                                              trainstate_val2_.params,
                                                              agent_for_state_history=2)

        aux1, aux2, aux_info = aux

        _, obs2_list, _, _, _, _, _, _ = aux2

        state_history_for_kl_div = unfinished_state_history
        state_history_for_kl_div.extend(obs2_list)

        other_pol_probs_ref = get_policies_for_states(subkey,
                                                      trainstate_th2_,
                                                      trainstate_th2_.params,
                                                      trainstate_val2_,
                                                      trainstate_val2_.params,
                                                      state_history_for_kl_div)
        other_state_history_ref = state_history_for_kl_div



    # preserving the params we want to diff through on the outer loop (th1)
    stuff = (reused_subkey, trainstate_th1, trainstate_th1_params,
             trainstate_val1, trainstate_val1_params,
             trainstate_th2_, trainstate_th2_.params,
             trainstate_val2_, trainstate_val2_.params, other_old_trainstate_th,
             other_old_trainstate_val, other_pol_probs_ref, other_state_history_ref)

    stuff, aux = inner_step_get_grad_otheragent2(stuff, None)

    _, _, _, _, _, trainstate_th2_, _, trainstate_val2_, _, _, _, _, _ = stuff

    key, subkey = jax.random.split(key)

    if args.inner_steps > 1:
        stuff = (subkey, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
                 trainstate_th2_, trainstate_th2_.params,
                 trainstate_val2_, trainstate_val2_.params,
                 other_old_trainstate_th, other_old_trainstate_val,
                 other_pol_probs_ref, other_state_history_ref)
        stuff, aux = jax.lax.scan(inner_step_get_grad_otheragent2, stuff,
                                  None, args.inner_steps - 1)
        _, _, _, _, _, trainstate_th2_, _, trainstate_val2_, _, _, _, _, _ = stuff

    if use_baseline:
        return trainstate_th2_, trainstate_val2_
    else:
        return trainstate_th2_, None


@jit
def inner_steps_plus_update_otheragent1(key, trainstate_th1, trainstate_th1_params,
                            trainstate_val1, trainstate_val1_params,
                            trainstate_th2, trainstate_th2_params,
                            trainstate_val2, trainstate_val2_params,
                            other_old_trainstate_th, other_old_trainstate_val):

    trainstate_th1_ = TrainState.create(apply_fn=trainstate_th1.apply_fn,
                                        params=trainstate_th1_params,
                                        tx=optax.sgd(
                                            learning_rate=args.lr_in))
    trainstate_val1_ = TrainState.create(apply_fn=trainstate_val1.apply_fn,
                                         params=trainstate_val1_params,
                                         tx=optax.sgd(
                                             learning_rate=args.lr_v))

    key, reused_subkey = jax.random.split(key)
    # reuse the subkey to get consistent trajectories for the first batch
    # This is only needed so I can be consistent with my previous pytorch code
    # And does not really have a theoretical or logical grounding really

    other_pol_probs_ref = None
    other_state_history_ref = None

    key, subkey = jax.random.split(key)

    if args.old_kl_div:
        stuff, aux, unfinished_state_history = do_env_rollout(reused_subkey,
                                                              trainstate_th1_,
                                                              trainstate_th1_.params,
                                                              trainstate_val1_,
                                                              trainstate_val1_.params,
                                                              trainstate_th2,
                                                              trainstate_th2_params,
                                                              trainstate_val2,
                                                              trainstate_val2_params,
                                                              agent_for_state_history=2)

        aux1, aux2, aux_info = aux

        _, obs1_list, _, _, _, _, _, _ = aux1

        state_history_for_kl_div = unfinished_state_history
        state_history_for_kl_div.extend(obs1_list)

        other_pol_probs_ref = get_policies_for_states(subkey,
                                                      trainstate_th1_,
                                                      trainstate_th1_.params,
                                                      trainstate_val1_,
                                                      trainstate_val1_.params,
                                                      state_history_for_kl_div)
        other_state_history_ref = state_history_for_kl_div

    # preserving the params we want to diff through on the outer loop (th2)
    stuff = (reused_subkey, trainstate_th1_, trainstate_th1_.params,
             trainstate_val1_, trainstate_val1_.params,
             trainstate_th2, trainstate_th2_params,
             trainstate_val2, trainstate_val2_params, other_old_trainstate_th,
             other_old_trainstate_val, other_pol_probs_ref, other_state_history_ref)

    stuff, aux = inner_step_get_grad_otheragent1(stuff, None)

    _, trainstate_th1_, _, trainstate_val1_, _, _, _, _, _, _, _, _, _ = stuff

    key, subkey = jax.random.split(key)

    if args.inner_steps > 1:
        stuff = (subkey, trainstate_th1_, trainstate_th1_.params, trainstate_val1_, trainstate_val1_.params,
                 trainstate_th2, trainstate_th2_params,
                 trainstate_val2, trainstate_val2_params,
                 other_old_trainstate_th, other_old_trainstate_val,
                 other_pol_probs_ref, other_state_history_ref)
        stuff, aux = jax.lax.scan(inner_step_get_grad_otheragent1, stuff,
                                  None, args.inner_steps - 1)
        _, trainstate_th1_, _, trainstate_val1_, _, _, _, _, _, _, _, _, _ = stuff

    if use_baseline:
        return trainstate_th1_, trainstate_val1_
    else:
        return trainstate_th1_, None

# @partial(jit, static_argnums=(11))
# # TODO can replace other_agent with each of the theta_p or theta_v or whatever. This could be one way to remove the agent class
# # So that things can be jittable. Everything has to be pure somehow
# def inner_steps_plus_update(key, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
#                             trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,
#                             other_old_trainstate_th, other_old_trainstate_val, other_agent=2 ):
#     # TODO ensure that this starts from scratch (well the agent 2 policy before the updates happen)
#     trainstate_th1_ = TrainState.create(apply_fn=trainstate_th1.apply_fn,
#                                         params=trainstate_th1_params,
#                                         tx=optax.sgd(
#                                             learning_rate=args.lr_in))
#     trainstate_val1_ = TrainState.create(apply_fn=trainstate_val1.apply_fn,
#                                          params=trainstate_val1_params,
#                                          tx=optax.sgd(
#                                              learning_rate=args.lr_v))
#
#     # TODO MAJOR TODO
#     # This is wrong isn't it, shouldn't be recreating both agents theta... only other agent's...
#     # And the optim too...
#     # Because the inner dice objective is going to be based on those params
#     # And you can't reset/create a new param otherwise the gradient won't flow
#     if other_agent == 2:
#
#
#         trainstate_th2_ = TrainState.create(apply_fn=trainstate_th2.apply_fn,
#                                              params=trainstate_th2_params,
#                                              tx=optax.sgd(
#                                                  learning_rate=args.lr_in))
#         trainstate_val2_ = TrainState.create(apply_fn=trainstate_val2.apply_fn,
#                                               params=trainstate_val2_params,
#                                               tx=optax.sgd(
#                                                   learning_rate=args.lr_v))
#
#
#     key, reused_subkey = jax.random.split(key)
#     # reuse the subkey to get consistent trajectories for the first batch
#     # This is only needed so I can be consistent with my previous pytorch code
#     # And does not really have a theoretical or logical grounding really
#
#     other_pol_probs_ref = None
#
#     key, subkey = jax.random.split(key)
#
#
#     if args.old_kl_div:
#         stuff, aux, unfinished_state_history = do_env_rollout(reused_subkey,
#                                                               trainstate_th1_,
#                                                               trainstate_th1_.params,
#                                                               trainstate_val1_,
#                                                               trainstate_val1_.params,
#                                                               trainstate_th2_,
#                                                               trainstate_th2_.params,
#                                                               trainstate_val2_,
#                                                               trainstate_val2_.params,
#                                                               agent_for_state_history=2)
#
#         aux1, aux2, aux_info = aux
#
#         if other_agent == 2:
#             _, obs2_list, _, _, _, _, _, _ = aux2
#
#             state_history_for_kl_div = unfinished_state_history
#             state_history_for_kl_div.extend(obs2_list)
#
#             other_pol_probs_ref = get_policies_for_states(subkey,
#                                                          trainstate_th2_,
#                                                          trainstate_th2_.params,
#                                                          trainstate_val2_,
#                                                          trainstate_val2_.params,
#                                                          state_history_for_kl_div)
#         else:
#             _, obs1_list, _, _, _, _, _, _ = aux1
#
#             state_history_for_kl_div = unfinished_state_history
#             state_history_for_kl_div.extend(obs1_list)
#
#             other_pol_probs_ref = get_policies_for_states(subkey,
#                                                           trainstate_th1_,
#                                                           trainstate_th1_.params,
#                                                           trainstate_val1_,
#                                                           trainstate_val1_.params,
#                                                           state_history_for_kl_div)
#
#     stuff = (reused_subkey, trainstate_th1_, trainstate_val1_, trainstate_th2_,
#              trainstate_val2_, other_old_trainstate_th, other_old_trainstate_val, other_pol_probs_ref)
#     if other_agent == 2:
#         stuff, aux = inner_step_get_grad_otheragent2(stuff, None)
#     else:
#         assert other_agent == 1
#         stuff, aux = inner_step_get_grad_otheragent1(stuff, None)
#
#     _, trainstate_th1_, trainstate_val1_, trainstate_th2_, trainstate_val2_, _, _, _ = stuff
#
#
#     # Somehow this loop is faster than the lax.scan?
#     # if args.inner_steps > 1:
#     # for inner_step in range(args.inner_steps-1):
#     #     subkey, subkey1 = jax.random.split(subkey)
#     #     stuff = (subkey1, trainstate_th1_, trainstate_val1_, trainstate_th2_,
#     #     trainstate_val2_, other_old_trainstate_th, other_old_trainstate_val)
#     #     if other_agent == 2:
#     #         stuff, aux  = inner_step_get_grad_otheragent2(stuff, None)
#     #     else:
#     #         stuff, aux  = inner_step_get_grad_otheragent1(stuff, None)
#     #     _, trainstate_th1_, trainstate_val1_, trainstate_th2_, trainstate_val2_, _, _ = stuff
#
#     key, subkey = jax.random.split(key)
#
#     if args.inner_steps > 1:
#         stuff = (subkey, trainstate_th1_, trainstate_val1_, trainstate_th2_,
#                  trainstate_val2_, other_old_trainstate_th,
#                  other_old_trainstate_val, other_pol_probs_ref)
#         if other_agent == 2:
#             stuff, aux = jax.lax.scan(inner_step_get_grad_otheragent2, stuff, None, args.inner_steps - 1)
#         else:
#             stuff, aux = jax.lax.scan(inner_step_get_grad_otheragent1, stuff, None, args.inner_steps - 1)
#         _, trainstate_th1_, trainstate_val1_, trainstate_th2_, trainstate_val2_, _, _, _ = stuff
#
#     ret_trainstate_val = None
#     if other_agent == 2:
#         ret_trainstate_th = trainstate_th2_
#         if use_baseline:
#             ret_trainstate_val = trainstate_val2_
#     else:
#         ret_trainstate_th = trainstate_th1_
#         if use_baseline:
#             ret_trainstate_val = trainstate_val1_
#
#     return ret_trainstate_th, ret_trainstate_val





@partial(jit, static_argnums=(11))
def out_lookahead(key, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
                  trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,
                  old_trainstate_th, old_trainstate_val, self_agent=1, self_pol_probs_ref=None, self_state_history_ref=None):

    stuff, aux, unfinished_state_history_for_kl_div = do_env_rollout(key, trainstate_th1,
                                                           trainstate_th1_params,
                                                           trainstate_val1,
                                                           trainstate_val1_params,
                                                           trainstate_th2,
                                                           trainstate_th2_params,
                                                           trainstate_val2,
                                                           trainstate_val2_params,
                                                           agent_for_state_history=self_agent)

    aux1, aux2, aux_info = aux
    state_history_for_kl_div = unfinished_state_history_for_kl_div

    # cat_act_probs_self = []

    key, env_state, obs1, obs2, \
    trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,\
    trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,\
    h_p1, h_v1, h_p2, h_v2 = stuff

    if self_agent == 1:
        cat_act_probs1_list, obs1_list, lp1_list, lp2_list, v1_list, r1_list, a1_list, a2_list = aux1

        # cat_act_probs_self.extend(cat_act_probs1_list)
        state_history_for_kl_div.extend(obs1_list)

        key, subkey = jax.random.split(key)
        # act just to get the final state values

        act_args1 = (subkey, obs1, trainstate_th1, trainstate_th1_params,
                     trainstate_val1, trainstate_val1_params, h_p1, h_v1)
        stuff1, aux1 = act(act_args1, None)
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1, logits1 = aux1

        # a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(subkey, obs1,
        #                                               trainstate_th1,
        #                                               trainstate_th1_params,
        #                                               trainstate_val1,
        #                                               trainstate_val1_params,
        #                                               h_p1, h_v1)

        end_state_v = v1
        objective = dice_objective_plus_value_loss(self_logprobs=lp1_list,
                                   other_logprobs=lp2_list,
                                   rewards=r1_list, values=v1_list,
                                   end_state_v=end_state_v)
        # print(f"Agent 1 episode return avg {r1_list.sum(axis=0).mean()}")
    else:
        assert self_agent == 2
        cat_act_probs2_list, obs2_list, lp2_list, lp1_list, v2_list, r2_list, a2_list, a1_list = aux2

        # cat_act_probs_self.extend(cat_act_probs2_list)
        state_history_for_kl_div.extend(obs2_list)

        key, subkey = jax.random.split(key)
        # act just to get the final state values
        act_args2 = (subkey, obs2, trainstate_th2, trainstate_th2_params,
                     trainstate_val2, trainstate_val2_params, h_p2, h_v2)
        stuff2, aux2 = act(act_args2, None)
        a2, lp2, v2, h_p2, h_v2, cat_act_probs2, logits2 = aux2
        # a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(subkey, obs2,
        #                                               trainstate_th2,
        #                                               trainstate_th2_params,
        #                                               trainstate_val2,
        #                                               trainstate_val2_params,
        #                                               h_p2, h_v2)
        end_state_v = v2
        objective = dice_objective_plus_value_loss(self_logprobs=lp2_list,
                                   other_logprobs=lp1_list,
                                   rewards=r2_list, values=v2_list,
                                   end_state_v=end_state_v)
        # print(f"Agent 2 episode return avg {r2_list.sum(axis=0).mean()}")

    key, sk1, sk2 = jax.random.split(key, 3)




    if args.old_kl_div:
        assert self_pol_probs_ref is not None # TODO save this somewhere, use the batch of states and pass it through once
        assert self_state_history_ref is not None
        if self_agent == 1:
            self_pol_probs = get_policies_for_states(sk1, trainstate_th1,
                                                     trainstate_th1_params,
                                                     trainstate_val1,
                                                     trainstate_val1_params,
                                                     self_state_history_ref)
        else:
            self_pol_probs = get_policies_for_states(sk1,
                                                     trainstate_th2,
                                                     trainstate_th2_params,
                                                     trainstate_val2,
                                                     trainstate_val2_params,
                                                     self_state_history_ref)
        # Then use this everywhere
    else:
        if self_agent == 1:
            self_pol_probs = get_policies_for_states(sk1, trainstate_th1,
                                                     trainstate_th1_params,
                                                     trainstate_val1,
                                                     trainstate_val1_params,
                                                     state_history_for_kl_div)
        else:
            self_pol_probs = get_policies_for_states(sk1,
                                                     trainstate_th2,
                                                     trainstate_th2_params,
                                                     trainstate_val2,
                                                     trainstate_val2_params,
                                                     state_history_for_kl_div)

        self_pol_probs_ref = get_policies_for_states(sk2,
                                                            old_trainstate_th,
                                                            old_trainstate_th.params,
                                                            old_trainstate_val,
                                                            old_trainstate_val.params,
                                                            state_history_for_kl_div)

    kl_div = kl_div_jax(self_pol_probs, self_pol_probs_ref)
    # print(f"Outer KL Div: {kl_div}")

    # return grad
    return objective + args.outer_beta * kl_div, state_history_for_kl_div


@partial(jit, static_argnums=(11))
def out_lookahead_no_kl(key, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
                  trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,
                  old_trainstate_th, old_trainstate_val, self_agent=1, self_pol_probs_ref=None, self_state_history_ref=None):

    stuff, aux, unfinished_state_history_for_kl_div = do_env_rollout(key, trainstate_th1,
                                                           trainstate_th1_params,
                                                           trainstate_val1,
                                                           trainstate_val1_params,
                                                           trainstate_th2,
                                                           trainstate_th2_params,
                                                           trainstate_val2,
                                                           trainstate_val2_params,
                                                           agent_for_state_history=self_agent)

    aux1, aux2, aux_info = aux
    state_history_for_kl_div = unfinished_state_history_for_kl_div

    # cat_act_probs_self = []

    key, env_state, obs1, obs2, \
    trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,\
    trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,\
    h_p1, h_v1, h_p2, h_v2 = stuff

    if self_agent == 1:
        cat_act_probs1_list, obs1_list, lp1_list, lp2_list, v1_list, r1_list, a1_list, a2_list = aux1

        # cat_act_probs_self.extend(cat_act_probs1_list)
        state_history_for_kl_div.extend(obs1_list)

        key, subkey = jax.random.split(key)
        # act just to get the final state values

        act_args1 = (subkey, obs1, trainstate_th1, trainstate_th1_params,
                     trainstate_val1, trainstate_val1_params, h_p1, h_v1)
        stuff1, aux1 = act(act_args1, None)
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1, logits1 = aux1

        # a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(subkey, obs1,
        #                                               trainstate_th1,
        #                                               trainstate_th1_params,
        #                                               trainstate_val1,
        #                                               trainstate_val1_params,
        #                                               h_p1, h_v1)

        end_state_v = v1
        objective = dice_objective_plus_value_loss(self_logprobs=lp1_list,
                                   other_logprobs=lp2_list,
                                   rewards=r1_list, values=v1_list,
                                   end_state_v=end_state_v)
        # print(f"Agent 1 episode return avg {r1_list.sum(axis=0).mean()}")
    else:
        assert self_agent == 2
        cat_act_probs2_list, obs2_list, lp2_list, lp1_list, v2_list, r2_list, a2_list, a1_list = aux2

        # cat_act_probs_self.extend(cat_act_probs2_list)
        state_history_for_kl_div.extend(obs2_list)

        key, subkey = jax.random.split(key)
        # act just to get the final state values
        act_args2 = (subkey, obs2, trainstate_th2, trainstate_th2_params,
                     trainstate_val2, trainstate_val2_params, h_p2, h_v2)
        stuff2, aux2 = act(act_args2, None)
        a2, lp2, v2, h_p2, h_v2, cat_act_probs2, logits2 = aux2
        # a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(subkey, obs2,
        #                                               trainstate_th2,
        #                                               trainstate_th2_params,
        #                                               trainstate_val2,
        #                                               trainstate_val2_params,
        #                                               h_p2, h_v2)
        end_state_v = v2
        objective = dice_objective_plus_value_loss(self_logprobs=lp2_list,
                                   other_logprobs=lp1_list,
                                   rewards=r2_list, values=v2_list,
                                   end_state_v=end_state_v)
        # print(f"Agent 2 episode return avg {r2_list.sum(axis=0).mean()}")

    key, sk1, sk2 = jax.random.split(key, 3)

    # return grad
    return objective, state_history_for_kl_div

@jit
def one_outer_step_objective_selfagent1(key, trainstate_th1_copy, trainstate_th1_copy_params, trainstate_val1_copy, trainstate_val1_copy_params,
                             trainstate_th2_copy, trainstate_th2_copy_params, trainstate_val2_copy, trainstate_val2_copy_params,
                             trainstate_th_ref, trainstate_val_ref, self_pol_probs_ref=None, self_state_history_ref=None):
    self_agent = 1
    other_agent = 2
    key, subkey = jax.random.split(key)
    trainstate_th2_after_inner_steps, trainstate_val2_after_inner_steps = \
        inner_steps_plus_update_otheragent2(subkey,
                                trainstate_th1_copy, trainstate_th1_copy_params,
                                trainstate_val1_copy,
                                trainstate_val1_copy_params,
                                trainstate_th2_copy, trainstate_th2_copy_params,
                                trainstate_val2_copy,
                                trainstate_val2_copy_params,
                                trainstate_th2_copy, trainstate_val2_copy
                                )

    # TODO perhaps one way to get around this conditional outer step with kind of global storage
    # is that we should have one call to the lookaheads in the first time step
    # and then a scan over the rest of the x steps

    # update own parameters from out_lookahead:

    if use_baseline:
        objective, state_hist_from_rollout = out_lookahead(key, trainstate_th1_copy,
                                  trainstate_th1_copy_params,
                                  trainstate_val1_copy,
                                  trainstate_val1_copy_params,
                                  trainstate_th2_after_inner_steps,
                                  trainstate_th2_after_inner_steps.params,
                                  trainstate_val2_after_inner_steps,
                                  trainstate_val2_after_inner_steps.params,
                                  trainstate_th_ref,
                                  trainstate_val_ref,
                                  self_agent=self_agent,
                                  self_pol_probs_ref=self_pol_probs_ref,
                                  self_state_history_ref=self_state_history_ref)
    else:
        objective, state_hist_from_rollout = out_lookahead(key, trainstate_th1_copy,
                                  trainstate_th1_copy_params,
                                  None, None,
                                  trainstate_th2_after_inner_steps,
                                  trainstate_th2_after_inner_steps.params,
                                  None, None,
                                  trainstate_th_ref,
                                  trainstate_val_ref,
                                  self_agent=self_agent,
                                  self_pol_probs_ref=self_pol_probs_ref,
                                  self_state_history_ref=self_state_history_ref)

    return objective, state_hist_from_rollout


@jit
def first_outer_step_objective_selfagent1(key, trainstate_th1_copy, trainstate_th1_copy_params, trainstate_val1_copy, trainstate_val1_copy_params,
                             trainstate_th2_copy, trainstate_th2_copy_params, trainstate_val2_copy, trainstate_val2_copy_params,
                             trainstate_th_ref, trainstate_val_ref):
    self_agent = 1
    other_agent = 2
    key, subkey = jax.random.split(key)
    trainstate_th2_after_inner_steps, trainstate_val2_after_inner_steps = \
        inner_steps_plus_update_otheragent2(subkey,
                                trainstate_th1_copy, trainstate_th1_copy_params,
                                trainstate_val1_copy,
                                trainstate_val1_copy_params,
                                trainstate_th2_copy, trainstate_th2_copy_params,
                                trainstate_val2_copy,
                                trainstate_val2_copy_params,
                                trainstate_th2_copy, trainstate_val2_copy
                                )

    # TODO perhaps one way to get around this conditional outer step with kind of global storage
    # is that we should have one call to the lookaheads in the first time step
    # and then a scan over the rest of the x steps

    # update own parameters from out_lookahead:

    if use_baseline:
        objective, state_hist_from_rollout = out_lookahead_no_kl(key, trainstate_th1_copy,
                                  trainstate_th1_copy_params,
                                  trainstate_val1_copy,
                                  trainstate_val1_copy_params,
                                  trainstate_th2_after_inner_steps,
                                  trainstate_th2_after_inner_steps.params,
                                  trainstate_val2_after_inner_steps,
                                  trainstate_val2_after_inner_steps.params,
                                  trainstate_th_ref,
                                  trainstate_val_ref,
                                  self_agent=self_agent,
                                  self_pol_probs_ref=None,
                                  self_state_history_ref=None)
    else:
        objective, state_hist_from_rollout = out_lookahead_no_kl(key, trainstate_th1_copy,
                                  trainstate_th1_copy_params,
                                  None, None,
                                  trainstate_th2_after_inner_steps,
                                  trainstate_th2_after_inner_steps.params,
                                  None, None,
                                  trainstate_th_ref,
                                  trainstate_val_ref,
                                  self_agent=self_agent,
                                  self_pol_probs_ref=None,
                                  self_state_history_ref=None)

    return objective, state_hist_from_rollout

@jit
def one_outer_step_objective_selfagent2(key, trainstate_th1_copy, trainstate_th1_copy_params, trainstate_val1_copy, trainstate_val1_copy_params,
                             trainstate_th2_copy, trainstate_th2_copy_params, trainstate_val2_copy, trainstate_val2_copy_params,
                             trainstate_th_ref, trainstate_val_ref, self_pol_probs_ref=None, self_state_history_ref=None):
    self_agent = 2
    other_agent = 1
    key, subkey = jax.random.split(key)
    trainstate_th1_after_inner_steps, trainstate_val1_after_inner_steps = \
        inner_steps_plus_update_otheragent1(subkey,
                                trainstate_th1_copy, trainstate_th1_copy_params,
                                trainstate_val1_copy,
                                trainstate_val1_copy_params,
                                trainstate_th2_copy, trainstate_th2_copy_params,
                                trainstate_val2_copy,
                                trainstate_val2_copy_params,
                                trainstate_th2_copy, trainstate_val2_copy)

    # TODO perhaps one way to get around this conditional outer step with kind of global storage
    # is that we should have one call to the lookaheads in the first time step
    # and then a scan over the rest of the x steps

    # update own parameters from out_lookahead:

    if use_baseline:
        objective, state_hist_from_rollout = out_lookahead(key, trainstate_th1_after_inner_steps,
                                  trainstate_th1_after_inner_steps.params,
                                  trainstate_val1_after_inner_steps,
                                  trainstate_val1_after_inner_steps.params,
                                  trainstate_th2_copy,
                                  trainstate_th2_copy_params,
                                  trainstate_val2_copy,
                                  trainstate_val2_copy.params,
                                  trainstate_th_ref,
                                  trainstate_val_ref,
                                  self_agent=self_agent,
                                  self_pol_probs_ref=self_pol_probs_ref,
                                  self_state_history_ref=self_state_history_ref)
    else:
        objective, state_hist_from_rollout = out_lookahead(key, trainstate_th1_after_inner_steps,
                                  trainstate_th1_after_inner_steps.params,
                                  None, None,
                                  trainstate_th2_copy,
                                  trainstate_th2_copy_params,
                                  None, None,
                                  trainstate_th_ref,
                                  trainstate_val_ref,
                                  self_agent=self_agent,
                                  self_pol_probs_ref=self_pol_probs_ref,
                                  self_state_history_ref=self_state_history_ref)

    return objective, state_hist_from_rollout

@jit
def first_outer_step_objective_selfagent2(key, trainstate_th1_copy, trainstate_th1_copy_params, trainstate_val1_copy, trainstate_val1_copy_params,
                             trainstate_th2_copy, trainstate_th2_copy_params, trainstate_val2_copy, trainstate_val2_copy_params,
                             trainstate_th_ref, trainstate_val_ref):
    self_agent = 2
    other_agent = 1
    key, subkey = jax.random.split(key)
    trainstate_th1_after_inner_steps, trainstate_val1_after_inner_steps = \
        inner_steps_plus_update_otheragent1(subkey,
                                trainstate_th1_copy, trainstate_th1_copy_params,
                                trainstate_val1_copy,
                                trainstate_val1_copy_params,
                                trainstate_th2_copy, trainstate_th2_copy_params,
                                trainstate_val2_copy,
                                trainstate_val2_copy_params,
                                trainstate_th2_copy, trainstate_val2_copy)

    # TODO perhaps one way to get around this conditional outer step with kind of global storage
    # is that we should have one call to the lookaheads in the first time step
    # and then a scan over the rest of the x steps

    # update own parameters from out_lookahead:

    if use_baseline:
        objective, state_hist_from_rollout = out_lookahead_no_kl(key, trainstate_th1_after_inner_steps,
                                  trainstate_th1_after_inner_steps.params,
                                  trainstate_val1_after_inner_steps,
                                  trainstate_val1_after_inner_steps.params,
                                  trainstate_th2_copy,
                                  trainstate_th2_copy_params,
                                  trainstate_val2_copy,
                                  trainstate_val2_copy.params,
                                  trainstate_th_ref,
                                  trainstate_val_ref,
                                  self_agent=self_agent,
                                  self_pol_probs_ref=None,
                                  self_state_history_ref=None)
    else:
        objective, state_hist_from_rollout = out_lookahead_no_kl(key, trainstate_th1_after_inner_steps,
                                  trainstate_th1_after_inner_steps.params,
                                  None, None,
                                  trainstate_th2_copy,
                                  trainstate_th2_copy_params,
                                  None, None,
                                  trainstate_th_ref,
                                  trainstate_val_ref,
                                  self_agent=self_agent,
                                  self_pol_probs_ref=None,
                                  self_state_history_ref=None)

    return objective, state_hist_from_rollout


@jit
def one_outer_step_update_selfagent1(stuff, unused):
    key, trainstate_th1_copy, trainstate_val1_copy, trainstate_th2_copy, trainstate_val2_copy, \
    trainstate_th_ref, trainstate_val_ref, self_pol_probs_ref, self_state_history_ref = stuff

    key, subkey = jax.random.split(key)

    obj_grad_fn = jax.grad(one_outer_step_objective_selfagent1, argnums=[2, 4], has_aux=True)

    (grad_th, grad_v), state_hist_from_rollout = obj_grad_fn(subkey,
                                  trainstate_th1_copy,
                                  trainstate_th1_copy.params,
                                  trainstate_val1_copy,
                                  trainstate_val1_copy.params,
                                  trainstate_th2_copy,
                                  trainstate_th2_copy.params,
                                  trainstate_val2_copy,
                                  trainstate_val2_copy.params,
                                  trainstate_th_ref, trainstate_val_ref,
                                  self_pol_probs_ref, self_state_history_ref)

    # update other's theta: NOTE HERE THIS IS JUST AN SGD UPDATE
    trainstate_th1_copy = trainstate_th1_copy.apply_gradients(grads=grad_th)

    # TODO when value update the inner model? Do it at all?
    if use_baseline:
        # Now this should be correct because I am using dice_objective_plus_value_loss
        # which has both the policy and the value loss together
        trainstate_val1_copy = trainstate_val1_copy.apply_gradients(grads=grad_v)

    # Since we only need the final trainstate, and not every trainstate every step of the way, no need for aux here
    stuff = (
    key, trainstate_th1_copy,  trainstate_val1_copy, trainstate_th2_copy,  trainstate_val2_copy,
    trainstate_th_ref, trainstate_val_ref, self_pol_probs_ref, self_state_history_ref)
    aux = state_hist_from_rollout

    return stuff, aux

@jit
def first_outer_step_update_selfagent1(stuff, unused):
    # Only for use with old kl div
    key, trainstate_th1_copy, trainstate_val1_copy, trainstate_th2_copy, trainstate_val2_copy, \
    trainstate_th_ref, trainstate_val_ref = stuff

    key, subkey = jax.random.split(key)

    obj_grad_fn = jax.grad(first_outer_step_objective_selfagent1, argnums=[2, 4], has_aux=True)

    (grad_th, grad_v), state_hist_from_rollout = obj_grad_fn(subkey,
                                  trainstate_th1_copy,
                                  trainstate_th1_copy.params,
                                  trainstate_val1_copy,
                                  trainstate_val1_copy.params,
                                  trainstate_th2_copy,
                                  trainstate_th2_copy.params,
                                  trainstate_val2_copy,
                                  trainstate_val2_copy.params,
                                  trainstate_th_ref, trainstate_val_ref,
                                  )

    # update other's theta: NOTE HERE THIS IS JUST AN SGD UPDATE
    trainstate_th1_copy = trainstate_th1_copy.apply_gradients(grads=grad_th)

    # TODO when value update the inner model? Do it at all?
    if use_baseline:
        # Now this should be correct because I am using dice_objective_plus_value_loss
        # which has both the policy and the value loss together
        trainstate_val1_copy = trainstate_val1_copy.apply_gradients(grads=grad_v)

    # Since we only need the final trainstate, and not every trainstate every step of the way, no need for aux here
    stuff = (
    key, trainstate_th1_copy,  trainstate_val1_copy, trainstate_th2_copy,  trainstate_val2_copy,
    trainstate_th_ref, trainstate_val_ref)
    aux = state_hist_from_rollout

    return stuff, aux

@jit
def one_outer_step_update_selfagent2(stuff, unused):
    key, trainstate_th1_copy, trainstate_val1_copy, \
    trainstate_th2_copy, trainstate_val2_copy,\
    trainstate_th_ref, trainstate_val_ref, self_pol_probs_ref, self_state_history_ref = stuff


    key, subkey = jax.random.split(key)

    obj_grad_fn = jax.grad(one_outer_step_objective_selfagent2, argnums=[6, 8], has_aux=True)

    (grad_th, grad_v), state_hist_from_rollout = obj_grad_fn(subkey,
                                  trainstate_th1_copy,
                                  trainstate_th1_copy.params,
                                  trainstate_val1_copy,
                                  trainstate_val1_copy.params,
                                  trainstate_th2_copy,
                                  trainstate_th2_copy.params,
                                  trainstate_val2_copy,
                                  trainstate_val2_copy.params,
                                  trainstate_th_ref, trainstate_val_ref,
                                  self_pol_probs_ref, self_state_history_ref)

    # update other's theta: NOTE HERE THIS IS JUST AN SGD UPDATE
    trainstate_th2_copy = trainstate_th2_copy.apply_gradients(grads=grad_th)

    # TODO when value update the inner model? Do it at all?
    if use_baseline:
        # Now this should be correct because I am using dice_objective_plus_value_loss
        # which has both the policy and the value loss together
        trainstate_val2_copy = trainstate_val2_copy.apply_gradients(grads=grad_v)

    # Since we only need the final trainstate, and not every trainstate every step of the way, no need for aux here
    stuff = (
    key, trainstate_th1_copy, trainstate_val1_copy,
    trainstate_th2_copy, trainstate_val2_copy,
    trainstate_th_ref, trainstate_val_ref, self_pol_probs_ref, self_state_history_ref)
    aux = state_hist_from_rollout

    return stuff, aux


@jit
def first_outer_step_update_selfagent2(stuff, unused):
    key, trainstate_th1_copy, trainstate_val1_copy, \
    trainstate_th2_copy, trainstate_val2_copy,\
    trainstate_th_ref, trainstate_val_ref = stuff


    key, subkey = jax.random.split(key)

    obj_grad_fn = jax.grad(first_outer_step_objective_selfagent2, argnums=[6, 8], has_aux=True)

    (grad_th, grad_v), state_hist_from_rollout = obj_grad_fn(subkey,
                                  trainstate_th1_copy,
                                  trainstate_th1_copy.params,
                                  trainstate_val1_copy,
                                  trainstate_val1_copy.params,
                                  trainstate_th2_copy,
                                  trainstate_th2_copy.params,
                                  trainstate_val2_copy,
                                  trainstate_val2_copy.params,
                                  trainstate_th_ref, trainstate_val_ref,
                                  )

    # update other's theta: NOTE HERE THIS IS JUST AN SGD UPDATE
    trainstate_th2_copy = trainstate_th2_copy.apply_gradients(grads=grad_th)

    # TODO when value update the inner model? Do it at all?
    if use_baseline:
        # Now this should be correct because I am using dice_objective_plus_value_loss
        # which has both the policy and the value loss together
        trainstate_val2_copy = trainstate_val2_copy.apply_gradients(grads=grad_v)

    # Since we only need the final trainstate, and not every trainstate every step of the way, no need for aux here
    stuff = (
    key, trainstate_th1_copy, trainstate_val1_copy,
    trainstate_th2_copy, trainstate_val2_copy,
    trainstate_th_ref, trainstate_val_ref)
    aux = state_hist_from_rollout

    return stuff, aux





# def rollout_collect_data_for_opp_model(self, other_theta_p, other_theta_v):
#     (s1, s2) = env.reset()
#     memory = Memory()
#     h_p1, h_v1, h_p2, h_v2 = (
#         jnp.zeros(args.batch_size, args.hidden_size).to(device),
#         jnp.zeros(args.batch_size, args.hidden_size).to(device),
#         jnp.zeros(args.batch_size, args.hidden_size).to(device),
#         jnp.zeros(args.batch_size, args.hidden_size).to(device))
#
#     state_history, other_state_history = [], []
#     state_history.append(s1)
#     other_state_history.append(s2)
#     act_history, other_act_history = [], []
#     other_rew_history = []
#
#
#     for t in range(args.rollout_len):
#         a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, self.theta_p,
#                                                       self.theta_v, h_p1,
#                                                       h_v1)
#         a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(s2, other_theta_p,
#                                                       other_theta_v, h_p2,
#                                                       h_v2)
#         (s1, s2), (r1, r2), _, _ = env.step((a1, a2))
#         memory.add(lp1, lp2, v1, r1)
#
#         state_history.append(s1)
#         other_state_history.append(s2)
#         act_history.append(a1)
#         other_act_history.append(a2)
#         other_rew_history.append(r2)
#
#     # Stacking dim = 0 gives (rollout_len, batch)
#     # Stacking dim = 1 gives (batch, rollout_len)
#     state_history = jnp.stack(state_history, dim=0)
#     other_state_history = jnp.stack(other_state_history, dim=0)
#     act_history = jnp.stack(act_history, dim=1)
#     other_act_history = jnp.stack(other_act_history, dim=1)
#
#     other_rew_history = jnp.stack(other_rew_history, dim=1)
#     return state_history, other_state_history, act_history, other_act_history, other_rew_history


# def opp_model(self, om_lr_p, om_lr_v, true_other_theta_p, true_other_theta_v,
#               prev_model_theta_p=None, prev_model_theta_v=None):
#     # true_other_theta_p and true_other_theta_v used only in the collection of data (rollouts in the environment)
#     # so then this is not cheating. We do not assume access to other agent policy parameters (at least not direct, white box access)
#     # We assume ability to collect trajectories through rollouts/play with the other agent in the environment
#     # Essentially when using OM, we are now no longer doing dice update on the trajectories collected directly (which requires parameter access)
#     # instead we collect the trajectories first, then build an OM, then rollout using OM and make DiCE/LOLA/POLA update based on that OM
#     # Instead of direct rollout using opponent true parameters and update based on that.
#     agent_opp = Agent(input_size, args.hidden_size, action_size, om_lr_p, om_lr_v, prev_model_theta_p, prev_model_theta_v)
#
#     opp_model_data_batches = args.opp_model_data_batches
#
#     for batch in range(opp_model_data_batches):
#         # should in principle only do 1 collect, but I can do multiple "batches"
#         # where repeating the below would be the same as collecting one big batch of environment interaction
#         state_history, other_state_history, act_history, other_act_history, other_rew_history =\
#             self.rollout_collect_data_for_opp_model(true_other_theta_p, true_other_theta_v)
#
#         opp_model_iters = 0
#         opp_model_steps_per_data_batch = args.opp_model_steps_per_batch
#
#         other_act_history = jnp.nn.functional.one_hot(other_act_history,
#                                                         action_size)
#
#         print(f"Opp Model Data Batch: {batch + 1}")
#
#         for opp_model_iter in range(opp_model_steps_per_data_batch):
#             # POLICY UPDATE
#             curr_pol_logits, curr_vals, final_state_vals = self.get_other_logits_values_for_states(agent_opp.theta_p,
#                                                                agent_opp.theta_v,
#                                                                other_state_history)
#
#
#             # KL div: p log p - p log q
#             # use p for target, since it has 0 and 1
#             # Then p log p has no deriv so can drop it, with respect to model
#             # then -p log q
#
#             # Calculate targets based on the action history (other act history)
#             # Essentially treat the one hot vector of actions as a class label, and then run supervised learning
#
#             c_e_loss = - (other_act_history * jnp.log_softmax(curr_pol_logits, dim=-1)).sum(dim=-1).mean()
#
#             print(c_e_loss.item())
#
#             agent_opp.theta_update(c_e_loss)
#
#             if use_baseline:
#                 # VALUE UPDATE
#                 v_loss = value_loss(values=curr_vals, rewards=other_rew_history, final_state_vals=final_state_vals)
#                 agent_opp.value_update(v_loss)
#
#             opp_model_iters += 1
#
#     return agent_opp.theta_p, agent_opp.theta_v

@jit
def eval_vs_alld_selfagent1(stuff, unused):
    key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v = stuff

    key, subkey = jax.random.split(key)

    act_args = (
    subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
    trainstate_val.params, h_p, h_v)

    stuff, aux = act(act_args, None)
    a, lp, v, h_p, h_v, cat_act_probs, logits = aux

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    i_am_red_agent = True
    opp_is_red_agent = False

    if args.env == "ipd":
        # Always defect
        a_opp = jnp.zeros_like(a)
    elif args.env == "coin":
        a_opp = env.get_moves_shortest_path_to_coin(env_state,
                                                    opp_is_red_agent)

    a1 = a
    a2 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2,
                                                          env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v)
    aux = (score1, score2)

    return stuff, aux


@jit
def eval_vs_alld_selfagent2(stuff, unused):
    key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v = stuff

    key, subkey = jax.random.split(key)

    act_args = (
    subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
    trainstate_val.params, h_p, h_v)

    stuff, aux = act(act_args, None)
    a, lp, v, h_p, h_v, cat_act_probs, logits = aux

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    i_am_red_agent = False
    opp_is_red_agent = True

    if args.env == "ipd":
        # Always defect
        a_opp = jnp.zeros_like(a)
    elif args.env == "coin":
        a_opp = env.get_moves_shortest_path_to_coin(env_state,
                                                    opp_is_red_agent)

    a2 = a
    a1 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2,
                                                          env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v)
    aux = (score1, score2)

    return stuff, aux

@jit
def eval_vs_allc_selfagent1(stuff, unused):
    key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v = stuff

    key, subkey = jax.random.split(key)

    act_args = (
    subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
    trainstate_val.params, h_p, h_v)

    stuff, aux = act(act_args, None)
    a, lp, v, h_p, h_v, cat_act_probs, logits = aux

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    i_am_red_agent = True
    opp_is_red_agent = False

    if args.env == "ipd":
        # Always cooperate
        a_opp = jnp.ones_like(a)
    elif args.env == "coin":
        a_opp = env.get_coop_action(env_state, opp_is_red_agent)

    a1 = a
    a2 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2,
                                                          env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v)
    aux = (score1, score2)

    return stuff, aux


@jit
def eval_vs_allc_selfagent2(stuff, unused):
    key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v = stuff

    key, subkey = jax.random.split(key)

    act_args = (
    subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
    trainstate_val.params, h_p, h_v)

    stuff, aux = act(act_args, None)
    a, lp, v, h_p, h_v, cat_act_probs, logits = aux

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    i_am_red_agent = False
    opp_is_red_agent = True

    if args.env == "ipd":
    # Always cooperate
        a_opp = jnp.ones_like(a)
    elif args.env == "coin":
        a_opp = env.get_coop_action(env_state, opp_is_red_agent)

    a2 = a
    a1 = a_opp

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2,
                                                          env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v)
    aux = (score1, score2)

    return stuff, aux


@jit
def eval_vs_tft_selfagent1(stuff, unused):
    key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v, prev_a, prev_agent_coin_collected_same_col, r1, r2 = stuff

    key, subkey = jax.random.split(key)

    act_args = (
    subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
    trainstate_val.params, h_p, h_v)

    stuff, aux = act(act_args, None)
    a, lp, v, h_p, h_v, cat_act_probs, logits = aux

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]



    if args.env == "ipd":
        # Copy last move of agent; assumes prev_a = all coop
        a_opp = prev_a
        prev_agent_coin_collected_same_col = None
    elif args.env == "coin":
        r_opp = r2
        # Agent here means me, the agent we are testing
        prev_agent_coin_collected_same_col = jnp.where(r_opp < 0, 0, prev_agent_coin_collected_same_col)
        prev_agent_coin_collected_same_col = jnp.where(r_opp > 0, 1, prev_agent_coin_collected_same_col)

        # prev_agent_coin_collected_same_col = prev_agent_coin_collected_same_col.at[
        #     r_opp < 0].set(0)  # opp got negative reward from other agent collecting opp's coin
        # prev_agent_coin_collected_same_col = prev_agent_coin_collected_same_col.at[
        #     r_opp > 0].set(1)  # opp is allowed to get positive reward from collecting own coin

        a_opp_defect = env.get_moves_shortest_path_to_coin(env_state, False)
        a_opp_coop = env.get_coop_action(env_state, False)

        a_opp = jax.lax.stop_gradient(a_opp_coop)
        # a_opp = a_opp.at[prev_agent_coin_collected_same_col == 0].set(a_opp_defect[prev_agent_coin_collected_same_col == 0])
        a_opp = jnp.where(prev_agent_coin_collected_same_col == 0, a_opp_defect, a_opp)

    a1 = a
    a2 = a_opp

    # print(env_state)
    # print(a1)
    # print(a2)
    # print(env_subkeys)

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2,
                                                          env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v, a, prev_agent_coin_collected_same_col, r1, r2)
    aux = (score1, score2)

    return stuff, aux


@jit
def eval_vs_tft_selfagent2(stuff, unused):
    key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v, prev_a, prev_agent_coin_collected_same_col, r1, r2 = stuff

    key, subkey = jax.random.split(key)

    act_args = (
    subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val,
    trainstate_val.params, h_p, h_v)

    stuff, aux = act(act_args, None)
    a, lp, v, h_p, h_v, cat_act_probs, logits = aux

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    if args.env == "ipd":
        # Copy last move of agent; assumes prev_a = all coop
        a_opp = prev_a
        prev_agent_coin_collected_same_col = None
    elif args.env == "coin":

        r_opp = r1
        # Agent here means me, the agent we are testing
        prev_agent_coin_collected_same_col = jnp.where(r_opp < 0, 0, prev_agent_coin_collected_same_col)
        prev_agent_coin_collected_same_col = jnp.where(r_opp > 0, 1, prev_agent_coin_collected_same_col)

        # prev_agent_coin_collected_same_col = prev_agent_coin_collected_same_col.at[
        #     r_opp < 0].set(0)  # opp got negative reward from other agent collecting opp's coin
        # prev_agent_coin_collected_same_col = prev_agent_coin_collected_same_col.at[
        #     r_opp > 0].set(1)  # opp is allowed to get positive reward from collecting own coin

        a_opp_defect = env.get_moves_shortest_path_to_coin(env_state, True)
        a_opp_coop = env.get_coop_action(env_state, True)

        a_opp = jax.lax.stop_gradient(a_opp_coop)
        # a_opp = a_opp.at[prev_agent_coin_collected_same_col == 0].set(a_opp_defect[prev_agent_coin_collected_same_col == 0])
        a_opp = jnp.where(prev_agent_coin_collected_same_col == 0, a_opp_defect, a_opp)

    a1 = a_opp
    a2 = a

    env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2,
                                                          env_subkeys)
    obsv = new_obs

    score1 = r1.mean()
    score2 = r2.mean()

    stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v, a, prev_agent_coin_collected_same_col, r1, r2)
    aux = (score1, score2)

    return stuff, aux



@partial(jit, static_argnums=(3, 4))
def eval_vs_fixed_strategy(key, trainstate_th, trainstate_val, strat="alld", self_agent=1):
    # if args.env == 'coin':
    #     print("eval vs fixed not yet implemented")
    #     return (None, None), None
    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    env_state, obsv = vec_env_reset(env_subkeys) # note this works only with the same obs, otherwise you would have to switch things up a bit here

    h_p = jnp.zeros((args.batch_size, args.hidden_size))
    h_v = None
    if use_baseline:
        h_v = jnp.zeros((args.batch_size, args.hidden_size))

    score1, score2 = 0., 0.

    if strat == "alld":
        stuff = key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v

        if self_agent == 1:
            stuff, aux = jax.lax.scan(eval_vs_alld_selfagent1, stuff, None, args.rollout_len)
        else:
            stuff, aux = jax.lax.scan(eval_vs_alld_selfagent2, stuff, None, args.rollout_len)
    elif strat == "allc":
        stuff = key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v

        if self_agent == 1:
            stuff, aux = jax.lax.scan(eval_vs_allc_selfagent1, stuff, None, args.rollout_len)
        else:
            stuff, aux = jax.lax.scan(eval_vs_allc_selfagent2, stuff, None, args.rollout_len)
    elif strat == "tft":
        if args.env == "ipd":
            prev_a = jnp.ones(
                args.batch_size, dtype=int)  # assume agent (self) cooperated for the init time step when the opponent is using TFT
            r1 = jnp.zeros(args.batch_size, dtype=int)  # these don't matter for IPD,
            r2 = jnp.zeros(args.batch_size, dtype=int)
            prev_agent_coin_collected_same_col = None
        elif args.env == "coin":
            if self_agent == 1:
                prev_a = env.get_coop_action(env_state,
                                             red_agent_perspective=False)  # doesn't matter for coin
            else:
                prev_a = env.get_coop_action(env_state,
                                             red_agent_perspective=True)  # doesn't matter for coin
            prev_agent_coin_collected_same_col = jnp.ones(
                args.batch_size, dtype=int)  # 0 = defect, collect other agent coin. Init with 1 (coop)
            r1 = jnp.zeros(args.batch_size, dtype=int)
            r2 = jnp.zeros(args.batch_size, dtype=int)
        else:
            raise NotImplementedError
        stuff = (
        key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v, prev_a,
        prev_agent_coin_collected_same_col, r1, r2)
        if self_agent == 1:
            stuff, aux = jax.lax.scan(eval_vs_tft_selfagent1, stuff, None,
                                      args.rollout_len)
        else:
            stuff, aux = jax.lax.scan(eval_vs_tft_selfagent2, stuff, None,
                                      args.rollout_len)

        # if self_agent == 1:
        #     # Opponent is Blue agent
        #     if args.env == "ipd":
        #         prev_a = jnp.ones(args.batch_size) # assume agent (self) cooperated for the init time step when the opponent is using TFT
        #         r1 = jnp.zeros(args.batch_size) # these don't matter for IPD,
        #         r2 = jnp.zeros(args.batch_size)
        #         prev_agent_coin_collected_same_col = None
        #     elif args.env == "coin":
        #         prev_a = env.get_coop_action(env_state, red_agent_perspective=False) # doesn't matter for coin
        #         prev_agent_coin_collected_same_col = jnp.ones(args.batch_size)  # 0 = defect, collect other agent coin. Init with 1 (coop)
        #         r1 = jnp.zeros(args.batch_size)
        #         r2 = jnp.zeros(args.batch_size)
        #     else:
        #         raise NotImplementedError
        #     stuff = (key, trainstate_th, trainstate_val, env_state, obsv, h_p, h_v, prev_a, prev_agent_coin_collected_same_col, r1, r2)
        #     stuff, aux = jax.lax.scan(eval_vs_tft_selfagent1, stuff, None, args.rollout_len)
        # else:
        #     a_opp = env.get_coop_action(env_state, red_agent_perspective=True)
        #     prev_agent_coin_collected_same_col = jnp.ones(args.batch_size)
        #     stuff, aux = jax.lax.scan(eval_vs_tft_selfagent2, stuff, None, args.rollout_len)

    score1, score2 = aux
    score1 = score1.mean()
    score2 = score2.mean()

    # TODO lax scan here... perhaps a separate lax scan for each strat
    # for t in range(args.rollout_len):
    #     if t > 0:
    #         prev_a = a
    #
    #     key, subkey = jax.random.split(key)
    #
    #     act_args = (subkey, obsv, trainstate_th, trainstate_th.params, trainstate_val, trainstate_val.params, h_p, h_v)
    #
    #     stuff, aux = act(act_args, None)
    #     a, lp, v, h_p, h_v, cat_act_probs, logits = aux
    #
    #     keys = jax.random.split(key, args.batch_size + 1)
    #     key, env_subkeys = keys[0], keys[1:]
    #
    #     if args.env == 'coin':
    #         if self_agent == 1:
    #             i_am_red_agent = True
    #             opp_is_red_agent = False
    #         else: # if I am agent 2 (blue), then opponent is agent 1 (red)
    #             i_am_red_agent = False
    #             opp_is_red_agent = True
    #
    #
    #     if strat == "alld":
    #         if args.env == "ipd":
    #             # Always defect
    #             a_opp = jnp.zeros_like(a)
    #         elif args.env == "coin":
    #             a_opp = env.get_moves_shortest_path_to_coin(env_state, opp_is_red_agent)
    #
    #         else:
    #             raise NotImplementedError
    #
    #     elif strat == "allc":
    #         if args.env == "ipd":
    #             # Always cooperate
    #             a_opp = jnp.ones_like(a)
    #         elif args.env == "coin":
    #             a_opp = env.get_coop_action(env_state, opp_is_red_agent)
    #         else:
    #             print("Eval vs fixed not yet implemented")
    #
    #     elif strat == "tft":
    #         if args.env == "ipd":
    #             if t == 0:
    #                 # start with coop
    #                 a_opp = jnp.ones_like(a)
    #             else:
    #                 # otherwise copy the last move of the other agent
    #                 a_opp = prev_a
    #         elif args.env == "coin":
    #             if t == 0:
    #                 a_opp = env.get_coop_action(env_state, opp_is_red_agent)
    #                 prev_agent_coin_collected_same_col = jnp.ones_like(a)  # 0 = defect, collect other agent coin
    #             else:
    #                 if i_am_red_agent:
    #                     r_opp = r2
    #                 else:
    #                     r_opp = r1
    #                 # Agent here means me, the agent we are testing
    #                 prev_agent_coin_collected_same_col = jnp.where(r_opp < 0, 0, prev_agent_coin_collected_same_col)
    #                 prev_agent_coin_collected_same_col = jnp.where(r_opp > 0, 1, prev_agent_coin_collected_same_col)
    #
    #                 # prev_agent_coin_collected_same_col = prev_agent_coin_collected_same_col.at[
    #                 #     r_opp < 0].set(0)  # opp got negative reward from other agent collecting opp's coin
    #                 # prev_agent_coin_collected_same_col = prev_agent_coin_collected_same_col.at[
    #                 #     r_opp > 0].set(1)  # opp is allowed to get positive reward from collecting own coin
    #
    #                 a_opp_defect = env.get_moves_shortest_path_to_coin(env_state, opp_is_red_agent)
    #                 a_opp_coop = env.get_coop_action(env_state, opp_is_red_agent)
    #
    #                 a_opp = jax.lax.stop_gradient(a_opp_coop)
    #                 # a_opp = a_opp.at[prev_agent_coin_collected_same_col == 0].set(a_opp_defect[prev_agent_coin_collected_same_col == 0])
    #                 a_opp = jnp.where(prev_agent_coin_collected_same_col == 0, a_opp_defect, a_opp)
    #         else:
    #             print("Eval vs fixed not yet implemented")
    #
    #     if self_agent == 1:
    #         a1 = a
    #         a2 = a_opp
    #     else:
    #         a1 = a_opp
    #         a2 = a
    #
    #     # print(strat)
    #     # print(env_state)
    #     # print(a1)
    #     # print(a2)
    #     # print(env_subkeys)
    #
    #     env_state, new_obs, (r1, r2), aux_info = vec_env_step(env_state, a1, a2, env_subkeys)
    #     obsv = new_obs
    #
    #     score1 += r1.mean()
    #     score2 += r2.mean()

    # score1 /= args.rollout_len
    # score2 /= args.rollout_len

    return (score1, score2), None

@jit
def get_init_hidden_states():
    h_p1, h_p2 = (
        jnp.zeros((args.batch_size, args.hidden_size)),
        jnp.zeros((args.batch_size, args.hidden_size))
    )
    h_v1, h_v2 = None, None
    if use_baseline:
        h_v1, h_v2 = (
            jnp.zeros((args.batch_size, args.hidden_size)),
            jnp.zeros((args.batch_size, args.hidden_size))
        )
    return h_p1, h_p2, h_v1, h_v2


def inspect_ipd(trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2):
    assert args.env == 'ipd'
    unused_keys = jax.random.split(jax.random.PRNGKey(0), args.batch_size)
    state, obsv = vec_env_reset(unused_keys)
    obs1 = obsv
    obs2 = obsv
    h_p1, h_p2, h_v1, h_v2 = get_init_hidden_states()
    # state_history = [obs1]
    #
    # stuff, aux = jax.lax.scan(env_step, stuff, None, 2)
    # aux1, aux2, aux_info = aux
    # cat_act_probs1_list, obs1_list, lp1_list, lp2_list, v1_list, r1_list, a1_list, a2_list = aux1
    #
    # state_history.extend(obs1_list)

    init_state = env.init_state

    for i in range(2):
        for j in range(2):
            state1 = env.states[i, j]
            for ii in range(2):
                for jj in range(2):
                    state2 = env.states[ii, jj]

                    state_history = [init_state, state1, state2]
                    print(state_history)

                    pol_probs1 = get_policies_for_states_onebatch(jax.random.PRNGKey(0),
                                                         trainstate_th1,
                                                         trainstate_th1.params,
                                                         trainstate_val1,
                                                         trainstate_val1.params,
                                                         state_history)
                    pol_probs2 = get_policies_for_states_onebatch(jax.random.PRNGKey(0),
                                                         trainstate_th2,
                                                         trainstate_th2.params,
                                                         trainstate_val2,
                                                         trainstate_val2.params,
                                                         state_history)
                    print(pol_probs1)
                    print(pol_probs2)



    # Build state history artificially for all combs, and pass those into the pol_probs.





@jit
def eval_progress(subkey, trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2):
    keys = jax.random.split(subkey, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]
    env_state, obsv = vec_env_reset(env_subkeys)
    obs1 = obsv
    obs2 = obsv
    h_p1, h_p2, h_v1, h_v2 = get_init_hidden_states()
    key, subkey = jax.random.split(key)
    stuff = (subkey, env_state, obs1, obs2,
             trainstate_th1, trainstate_th1.params, trainstate_val1,
             trainstate_val1.params,
             trainstate_th2, trainstate_th2.params, trainstate_val2,
             trainstate_val2.params,
             h_p1, h_v1, h_p2, h_v2)

    stuff, aux = jax.lax.scan(env_step, stuff, None, args.rollout_len)
    aux1, aux2, aux_info = aux

    # cat_act_probs1, obs1, lp1, lp2, v1, r1, a1, a2 = aux1
    # _, _, _, _, _, r2, _, _ = aux2
    #
    # for i in range(args.rollout_len):
    #     print(f"step {i}")
    #     print(obs1[i])
    #     print(a1[i+1])
    #     print(a2[i+1])
    #     print(r1[i+1])
    #     print(r2[i+1])
    #     print(obs1[i+1])

    _, _, _, _, _, r1, _, _ = aux1
    _, _, _, _, _, r2, _, _ = aux2

    score1rec = []
    score2rec = []

    # if args.env == 'coin':
    #     print("Eval vs Fixed strats not yet implemented")
    # else:
    print("Eval vs Fixed Strategies:")
    for strat in ["alld", "allc", "tft"]:
        # print(f"Playing against strategy: {strat.upper()}")
        key, subkey = jax.random.split(key)
        score1, _ = eval_vs_fixed_strategy(subkey, trainstate_th1, trainstate_val1, strat, self_agent=1)
        score1rec.append(score1[0])
        # print(f"Agent 1 score: {score1[0]}")
        key, subkey = jax.random.split(key)
        score2, _ = eval_vs_fixed_strategy(subkey, trainstate_th2, trainstate_val2, strat, self_agent=2)
        score2rec.append(score2[1])
        # print(f"Agent 2 score: {score2[1]}")

        # print(score1)
        # print(score2)

    score1rec = jnp.stack(score1rec)
    score2rec = jnp.stack(score2rec)

    # TODO the score in the old file had another division by the rollout length
    # But score is only used for eval and doesn't affect gradients
    # But KL Div definitely would affect the gradients.
    # avg_rew1 = r1.sum(axis=0).mean()
    avg_rew1 = r1.mean()
    # avg_rew2 = r2.sum(axis=0).mean()
    avg_rew2 = r2.mean()

    if args.env == 'coin':
        rr_matches, rb_matches, br_matches, bb_matches = aux_info
        rr_matches_amount = rr_matches.sum(axis=0).mean()
        rb_matches_amount = rb_matches.sum(axis=0).mean()
        br_matches_amount = br_matches.sum(axis=0).mean()
        bb_matches_amount = bb_matches.sum(axis=0).mean()
        return avg_rew1, avg_rew2, rr_matches_amount, rb_matches_amount, br_matches_amount, bb_matches_amount, score1rec, score2rec

    else:
        return avg_rew1, avg_rew2, None, None, None, None, score1rec, score2rec







def play(key, init_trainstate_th1, init_trainstate_val1, init_trainstate_th2, init_trainstate_val2,
         # theta_v1, theta_v1_params, agent1_value_optimizer,
         # theta_p2, theta_p2_params, agent2_theta_optimizer,
         # theta_v2, theta_v2_params, agent2_value_optimizer,
         use_opp_model=False): #,prev_scores=None, prev_coins_collected_info=None):
    joint_scores = []
    score_record = []
    # You could do something like the below and then modify the code to just be one continuous record that includes past values when loading from checkpoint
    # if prev_scores is not None:
    #     score_record = prev_scores
    # I'm tired though.
    vs_fixed_strats_score_record = [[], []]

    print("start iterations with", args.inner_steps, "inner steps and", args.outer_steps, "outer steps:")
    same_colour_coins_record = []
    diff_colour_coins_record = []
    coins_collected_info = (same_colour_coins_record, diff_colour_coins_record)

    agent2_theta_p_model, agent1_theta_p_model = None, None
    agent2_theta_v_model, agent1_theta_v_model = None, None

    # Pretty sure this creation is unnecessary and we can directly use the trainstates passed in
    trainstate_th1 = TrainState.create(apply_fn=init_trainstate_th1.apply_fn,
                                       params=init_trainstate_th1.params,
                                       tx=init_trainstate_th1.tx)
    trainstate_val1 = TrainState.create(apply_fn=init_trainstate_val1.apply_fn,
                                        params=init_trainstate_val1.params,
                                        tx=init_trainstate_val1.tx)
    trainstate_th2 = TrainState.create(apply_fn=init_trainstate_th2.apply_fn,
                                       params=init_trainstate_th2.params,
                                       tx=init_trainstate_th2.tx)
    trainstate_val2 = TrainState.create(apply_fn=init_trainstate_val2.apply_fn,
                                        params=init_trainstate_val2.params,
                                        tx=init_trainstate_val2.tx)

    # key, subkey = jax.random.split(key)
    # score1, score2, rr_matches_amount, rb_matches_amount, br_matches_amount, bb_matches_amount = \
    #     eval_progress(key, trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2)


    for update in range(args.n_update):
        # TODO RECREATE TRAINSTATES IF NEEDED

        trainstate_th1_copy = TrainState.create(
            apply_fn=trainstate_th1.apply_fn,
            params=trainstate_th1.params,
            tx=trainstate_th1.tx)
        trainstate_val1_copy = TrainState.create(
            apply_fn=trainstate_val1.apply_fn,
            params=trainstate_val1.params,
            tx=trainstate_val1.tx)
        trainstate_th2_copy = TrainState.create(
            apply_fn=trainstate_th2.apply_fn,
            params=trainstate_th2.params,
            tx=trainstate_th2.tx)
        trainstate_val2_copy = TrainState.create(
            apply_fn=trainstate_val2.apply_fn,
            params=trainstate_val2.params,
            tx=trainstate_val2.tx)

        # TODO there may be redundancy here, clean up later
        # THESE SHOULD NOT BE UPDATED (they are reset only on each new update step e.g. epoch, after all the outer and inner steps)
        trainstate_th1_ref = TrainState.create(
            apply_fn=trainstate_th1.apply_fn,
            params=trainstate_th1.params,
            tx=trainstate_th1.tx)
        trainstate_val1_ref = TrainState.create(
            apply_fn=trainstate_val1.apply_fn,
            params=trainstate_val1.params,
            tx=trainstate_val1.tx)
        trainstate_th2_ref = TrainState.create(
            apply_fn=trainstate_th2.apply_fn,
            params=trainstate_th2.params,
            tx=trainstate_th2.tx)
        trainstate_val2_ref = TrainState.create(
            apply_fn=trainstate_val2.apply_fn,
            params=trainstate_val2.params,
            tx=trainstate_val2.tx)

        # if use_opp_model:
        #     agent2_theta_p_model, agent2_theta_v_model = agent1.opp_model(args.om_lr_p, args.om_lr_v,
        #                                             true_other_theta_p=start_theta2,
        #                                             true_other_theta_v=start_val2,
        #                                             prev_model_theta_p=agent2_theta_p_model)
        #     agent1_theta_p_model, agent1_theta_v_model = agent2.opp_model(args.om_lr_p, args.om_lr_v,
        #                                             true_other_theta_p=start_theta1,
        #                                             true_other_theta_v=start_val1,
        #                                             prev_model_theta_p=agent1_theta_p_model)

        agent1_copy_for_val_update = None
        agent2_copy_for_val_update = None
        # if args.val_update_after_loop:
        #     theta_p_1_copy_for_vals = [tp.detach().clone().requires_grad_(True) for tp in
        #                     agent1.theta_p]
        #     theta_v_1_copy_for_vals = [tv.detach().clone().requires_grad_(True) for tv in
        #                   agent1.theta_v]
        #     theta_p_2_copy_for_vals = [tp.detach().clone().requires_grad_(True) for tp in
        #                     agent2.theta_p]
        #     theta_v_2_copy_for_vals = [tv.detach().clone().requires_grad_(True) for tv in
        #                   agent2.theta_v]
        #     agent1_copy_for_val_update = Agent(input_size, args.hidden_size,
        #                                       action_size, args.lr_out,
        #                                       args.lr_v, theta_p_1_copy_for_vals,
        #                                       theta_v_1_copy_for_vals)
        #     agent2_copy_for_val_update = Agent(input_size, args.hidden_size,
        #                                       action_size, args.lr_out,
        #                                       args.lr_v, theta_p_2_copy_for_vals,
        #                                       theta_v_2_copy_for_vals)

        key, subkey = jax.random.split(key)
        # reuse the subkey to get consistent trajectories for the first batch
        # This is only needed so I can be consistent with my previous pytorch code
        # And does not really have a theoretical or logical grounding really

        # key, env_state, obs1, obs2, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params, \
        # trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params, h_p1, h_v1, h_p2, h_v2 = stuff

        self_pol_probs_ref = None
        self_state_history_ref = None
        # TODO use the do_env_rollout and save self_pol_prob somewhere
        # TODO then do the same for agent 2.
        if args.old_kl_div:

            stuff = (subkey, trainstate_th1_copy, trainstate_val1_copy,
                     trainstate_th2_copy, trainstate_val2_copy,
                     trainstate_th1_ref, trainstate_val1_ref,
                     )

            stuff, aux = first_outer_step_update_selfagent1(stuff, None)
            _, trainstate_th1_copy, trainstate_val1_copy, _, _, _, _ = stuff
            self_state_history_ref = aux
            key, subkey = jax.random.split(key)
            self_pol_probs_ref = jax.lax.stop_gradient(
                get_policies_for_states(subkey,
                                        trainstate_th1_ref,
                                        trainstate_th1_ref.params,
                                        trainstate_val1_ref,
                                        trainstate_val1_ref.params,
                                        self_state_history_ref))

            if args.outer_steps > 1:
                key, subkey = jax.random.split(key)
                stuff = (subkey, trainstate_th1_copy, trainstate_val1_copy,
                         trainstate_th2_copy, trainstate_val2_copy,
                         trainstate_th1_ref, trainstate_val1_ref,
                         self_pol_probs_ref, self_state_history_ref)

                stuff, aux = jax.lax.scan(one_outer_step_update_selfagent1,
                                          stuff, None, args.outer_steps - 1)
                _, trainstate_th1_copy, trainstate_val1_copy, _, _, _, _, _, _ = stuff


        else:
            key, subkey = jax.random.split(key)
            stuff = (subkey, trainstate_th1_copy, trainstate_val1_copy,
                     trainstate_th2_copy, trainstate_val2_copy,
                     trainstate_th1_ref, trainstate_val1_ref, self_pol_probs_ref, self_state_history_ref)

            stuff, aux = jax.lax.scan(one_outer_step_update_selfagent1, stuff, None, args.outer_steps)
            _, trainstate_th1_copy, trainstate_val1_copy, _, _, _, _, _, _ = stuff

        # Doing this just as a safety failcase scenario, and copy this at the end
        trainstate_after_outer_steps_th1 = TrainState.create(
            apply_fn=trainstate_th1_copy.apply_fn,
            params=trainstate_th1_copy.params,
            tx=trainstate_th1_copy.tx)
        trainstate_after_outer_steps_val1 = TrainState.create(
            apply_fn=trainstate_val1_copy.apply_fn,
            params=trainstate_val1_copy.params,
            tx=trainstate_val1_copy.tx)

        # --- START OF AGENT 2 UPDATE ---

        # Doing this just as a safety failcase scenario, to make sure each agent loop starts from the beginning
        trainstate_th1_copy = TrainState.create(
            apply_fn=trainstate_th1.apply_fn,
            params=trainstate_th1.params,
            tx=trainstate_th1.tx)
        trainstate_val1_copy = TrainState.create(
            apply_fn=trainstate_val1.apply_fn,
            params=trainstate_val1.params,
            tx=trainstate_val1.tx)
        trainstate_th2_copy = TrainState.create(
            apply_fn=trainstate_th2.apply_fn,
            params=trainstate_th2.params,
            tx=trainstate_th2.tx)
        trainstate_val2_copy = TrainState.create(
            apply_fn=trainstate_val2.apply_fn,
            params=trainstate_val2.params,
            tx=trainstate_val2.tx)
        # TODO essentially what I think I need is to have the out lookahead include the in lookahead component... or a new outer step function
        # kind of like what I have in the other file, where then I can use that outer step function to take the grad of agent 1 params after
        # agent 2 has done a bunch of inner steps.

        key, reused_subkey = jax.random.split(key)
        # reuse the subkey to get consistent trajectories for the first batch
        # This is only needed so I can be consistent with my previous pytorch code
        # And does not really have a theoretical or logical grounding really

        # key, env_state, obs1, obs2, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params, \
        # trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params, h_p1, h_v1, h_p2, h_v2 = stuff

        self_pol_probs_ref = None
        self_state_history_ref = None

        if args.old_kl_div:

            stuff = (subkey, trainstate_th1_copy, trainstate_val1_copy,
                     trainstate_th2_copy, trainstate_val2_copy,
                     trainstate_th2_ref, trainstate_val2_ref,
                     )

            stuff, aux = first_outer_step_update_selfagent2(stuff, None)
            _, _, _, trainstate_th2_copy, trainstate_val2_copy, _, _ = stuff
            self_state_history_ref = aux
            key, subkey = jax.random.split(key)
            self_pol_probs_ref = jax.lax.stop_gradient(
                get_policies_for_states(subkey,
                                        trainstate_th2_ref,
                                        trainstate_th2_ref.params,
                                        trainstate_val2_ref,
                                        trainstate_val2_ref.params,
                                        self_state_history_ref))

            if args.outer_steps > 1:
                key, subkey = jax.random.split(key)
                stuff = (subkey, trainstate_th1_copy, trainstate_val1_copy,
                         trainstate_th2_copy, trainstate_val2_copy,
                         trainstate_th2_ref, trainstate_val2_ref,
                         self_pol_probs_ref, self_state_history_ref)

                stuff, aux = jax.lax.scan(one_outer_step_update_selfagent2,
                                          stuff, None, args.outer_steps - 1)
                _, _, _, trainstate_th2_copy, trainstate_val2_copy, _, _, _, _ = stuff


        else:

        # if args.old_kl_div:
        #     stuff, aux, unfinished_state_history = do_env_rollout(reused_subkey,
        #                                                           trainstate_th1_copy,
        #                                                           trainstate_th1_copy.params,
        #                                                           trainstate_val1_copy,
        #                                                           trainstate_val1_copy.params,
        #                                                           trainstate_th2_copy,
        #                                                           trainstate_th2_copy.params,
        #                                                           trainstate_val2_copy,
        #                                                           trainstate_val2_copy.params,
        #                                                           agent_for_state_history=2)
        #
        #     aux1, aux2, aux_info = aux
        #     _, obs2_list, _, _, _, _, _, _ = aux2
        #
        #     state_history_for_kl_div = unfinished_state_history
        #     state_history_for_kl_div.extend(obs2_list)
        #
        #     key, subkey = jax.random.split(key)
        #
        #     self_pol_probs_ref = jax.lax.stop_gradient(get_policies_for_states(subkey,
        #                                                  trainstate_th2_copy,
        #                                                  trainstate_th2_copy.params,
        #                                                  trainstate_val2_copy,
        #                                                  trainstate_val2_copy.params,
        #                                                  state_history_for_kl_div))
        #     self_state_history_ref = state_history_for_kl_div

            key, subkey = jax.random.split(key)

            stuff = (subkey, trainstate_th1_copy, trainstate_val1_copy,
                     trainstate_th2_copy, trainstate_val2_copy,
                     trainstate_th2_ref, trainstate_val2_ref,
                     self_pol_probs_ref, self_state_history_ref)

            stuff, aux = jax.lax.scan(one_outer_step_update_selfagent2, stuff, None,
                                      args.outer_steps)
            _, _, _, trainstate_th2_copy, trainstate_val2_copy, _, _, _, _ = stuff

        trainstate_after_outer_steps_th2 = TrainState.create(
            apply_fn=trainstate_th2_copy.apply_fn,
            params=trainstate_th2_copy.params,
            tx=trainstate_th2_copy.tx)
        trainstate_after_outer_steps_val2 = TrainState.create(
            apply_fn=trainstate_val2_copy.apply_fn,
            params=trainstate_val2_copy.params,
            tx=trainstate_val2_copy.tx)


        # TODO ensure this is correct. Ensure that the copy is updated on the outer loop once that has finished.
        # Note that this is updated only after all the outer loop steps have finished. the copies are
        # updated during the outer loops. But the main trainstate (like the main th) is updated only
        # after the loops finish
        trainstate_th1 = trainstate_after_outer_steps_th1
        trainstate_th2 = trainstate_after_outer_steps_th2

        trainstate_val1 = trainstate_after_outer_steps_val1
        trainstate_val2 = trainstate_after_outer_steps_val2

        # if args.val_update_after_loop:
        #     updated_theta_v_1 = [tv.detach().clone().requires_grad_(True)
        #                                for tv in
        #                                agent1_copy_for_val_update.theta_v]
        #     updated_theta_v_2 = [tv.detach().clone().requires_grad_(True)
        #                                for tv in
        #                                agent2_copy_for_val_update.theta_v]
        #     agent1.theta_v = updated_theta_v_1
        #     agent2.theta_v = updated_theta_v_2

        # evaluate progress:
        key, subkey = jax.random.split(key)
        score1, score2, rr_matches_amount, rb_matches_amount, br_matches_amount, bb_matches_amount, score1rec, score2rec = \
            eval_progress(key, trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2)



        # keys = jax.random.split(key, args.batch_size + 1)
        # key, env_subkeys = keys[0], keys[1:]
        # env_state, obsv = vec_env_reset(env_subkeys)
        # obs1 = obsv
        # obs2 = obsv
        # h_p1, h_p2 = (
        #     jnp.zeros((args.batch_size, args.hidden_size)),
        #     jnp.zeros((args.batch_size, args.hidden_size))
        # )
        # h_v1, h_v2 = None, None
        # if use_baseline:
        #     h_v1, h_v2 = (
        #         jnp.zeros((args.batch_size, args.hidden_size)),
        #         jnp.zeros((args.batch_size, args.hidden_size))
        #     )
        # key, subkey = jax.random.split(key)
        # stuff = (subkey, env_state, obs1, obs2,
        #          trainstate_th1, trainstate_th1.params, trainstate_val1,
        #          trainstate_val1.params,
        #          trainstate_th2, trainstate_th2.params, trainstate_val2,
        #          trainstate_val2.params,
        #          h_p1, h_v1, h_p2, h_v2)
        #
        # stuff, aux = jax.lax.scan(env_step, stuff, None, args.rollout_len)
        # aux1, aux2, aux_info = aux
        #
        # _, _, _, _, _, r1, _, _ = aux1
        # _, _, _, _, _, r2, _, _ = aux2
        #
        # rr_matches, rb_matches, br_matches, bb_matches = aux_info
        #
        # score1 = r1.sum(axis=0).mean()
        # score2 = r2.sum(axis=0).mean()
        # # print("Reward info (avg over batch, sum over episode length)")
        # # print(score1)
        # # print(score2)
        #
        # rr_matches_amount = rr_matches.sum(axis=0).mean()
        # rb_matches_amount = rb_matches.sum(axis=0).mean()
        # br_matches_amount = br_matches.sum(axis=0).mean()
        # bb_matches_amount = bb_matches.sum(axis=0).mean()

        # print("Matched coin info (avg over batch, sum over episode length)")
        # print(rr_matches_amount)
        # print(rb_matches_amount)
        # print(br_matches_amount)
        # print(bb_matches_amount)



        if args.env == "coin":
            same_colour_coins = rr_matches_amount + bb_matches_amount
            diff_colour_coins = rb_matches_amount + br_matches_amount
            same_colour_coins_record.append(same_colour_coins)
            diff_colour_coins_record.append(diff_colour_coins)

        vs_fixed_strats_score_record[0].append(score1rec)
        vs_fixed_strats_score_record[1].append(score2rec)

        score_record.append(jnp.stack((score1, score2)))

        # print
        if update % args.print_every == 0:
            print("*" * 10)
            print("Epoch: {}".format(update + 1), flush=True)
            print(f"Score for Agent 1: {score1}")
            print(f"Score for Agent 2: {score2}")
            if args.env == 'coin':
                print("Same coins: {}".format(rr_matches_amount + bb_matches_amount))
                print("Diff coins: {}".format(rb_matches_amount + br_matches_amount))
                print("RR coins {}".format(rr_matches_amount))
                print("RB coins {}".format(rb_matches_amount))
                print("BR coins {}".format(br_matches_amount))
                print("BB coins {}".format(bb_matches_amount))

            print("Scores vs fixed strats ALLD, ALLC, TFT:")
            print(score1rec)
            print(score2rec)

            if args.env == 'ipd':
                # print("Scores vs fixed strats ALLD, ALLC, TFT:")
                # print(score1rec)
                # print(score2rec)
                if args.inspect_ipd:
                    inspect_ipd(trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2)

        if update % args.checkpoint_every == 0:
            now = datetime.datetime.now()

            # print(coins_collected_info)
            # print(score_record)
            # print(vs_fixed_strats_score_record)

            checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
                                        target=(trainstate_th1, trainstate_val1,
                                                trainstate_th2, trainstate_val2,
                                                coins_collected_info,
                                                score_record,
                                                vs_fixed_strats_score_record),
                                        step=update + 1, prefix=f"checkpoint_{now.strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_epoch")


    return joint_scores




if __name__ == "__main__":
    parser = argparse.ArgumentParser("POLA")
    parser.add_argument("--inner_steps", type=int, default=1, help="inner loop steps for DiCE")
    parser.add_argument("--outer_steps", type=int, default=1, help="outer loop steps for POLA")
    parser.add_argument("--lr_out", type=float, default=0.005,
                        help="outer loop learning rate: same learning rate across all policies for now")
    parser.add_argument("--lr_in", type=float, default=0.03,
                        help="inner loop learning rate (eta): this has no use in the naive learning case. Used for the gradient step done for the lookahead for other agents during LOLA (therefore, often scaled to be higher than the outer learning rate in non-proximal LOLA). Note that this has a different meaning for the Taylor approx vs. actual update versions. A value of eta=1 is perfectly reasonable for the Taylor approx version as this balances the scale of the gradient with the naive learning term (and will be multiplied by the outer learning rate after), whereas for the actual update version with neural net, 1 is way too big an inner learning rate. For prox, this is the learning rate on the inner prox loop so is not that important - you want big enough to be fast-ish, but small enough to converge.")
    parser.add_argument("--lr_v", type=float, default=0.001,
                        help="same learning rate across all policies for now. Should be around maybe 0.001 or less for neural nets to avoid instability")
    parser.add_argument("--gamma", type=float, default=0.96, help="discount rate")
    parser.add_argument("--n_update", type=int, default=5000, help="number of epochs to run")
    parser.add_argument("--rollout_len", type=int, default=50, help="How long we want the time horizon of the game to be (number of steps before termination/number of iterations of the IPD)")
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1, help="for seed")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--print_every", type=int, default=1, help="Print every x number of epochs")
    parser.add_argument("--outer_beta", type=float, default=0.0, help="for outer kl penalty with POLA")
    parser.add_argument("--inner_beta", type=float, default=0.0, help="for inner kl penalty with POLA")
    parser.add_argument("--save_dir", type=str, default='.', help="Where to save checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=50, help="Epochs between checkpoint save")
    parser.add_argument("--load_dir", type=str, default=None, help="Directory for loading checkpoint")
    parser.add_argument("--load_prefix", type=str, default=None, help="Prefix for loading checkpoint")
    # parser.add_argument("--ent_reg", type=float, default=0.0, help="entropy regularizer")
    parser.add_argument("--diff_coin_reward", type=float, default=1.0, help="changes problem setting (the reward for picking up coin of different colour)")
    parser.add_argument("--diff_coin_cost", type=float, default=-2.0, help="changes problem setting (the cost to the opponent when you pick up a coin of their colour)")
    parser.add_argument("--same_coin_reward", type=float, default=1.0, help="changes problem setting (the reward for picking up coin of same colour)")
    parser.add_argument("--grid_size", type=int, default=3, help="Grid size for Coin Game")
    parser.add_argument("--optim", type=str, default="adam", help="Used only for the outer agent (in the out_lookahead)")
    parser.add_argument("--no_baseline", action="store_true", help="Use NO Baseline (critic) for variance reduction. Default is baseline using Loaded DiCE with GAE")
    parser.add_argument("--opp_model", action="store_true", help="Use Opponent Modeling")
    parser.add_argument("--opp_model_steps_per_batch", type=int, default=1, help="How many steps to train opp model on each batch at the beginning of each POLA epoch")
    parser.add_argument("--opp_model_data_batches", type=int, default=100, help="How many batches of data (right now from rollouts) to train opp model on")
    parser.add_argument("--om_lr_p", type=float, default=0.005,
                        help="learning rate for opponent modeling (imitation/supervised learning) for policy")
    parser.add_argument("--om_lr_v", type=float, default=0.001,
                        help="learning rate for opponent modeling (imitation/supervised learning) for value")
    parser.add_argument("--env", type=str, default="coin",
                        choices=["ipd", "coin"])
    parser.add_argument("--hist_one", action="store_true", help="Use one step history (no gru or rnn, just one step history)")
    parser.add_argument("--print_info_each_outer_step", action="store_true", help="For debugging/curiosity sake")
    parser.add_argument("--init_state_coop", action="store_true", help="For IPD only: have the first state be CC instead of a separate start state")
    parser.add_argument("--split_coins", action="store_true", help="If true, then when both agents step on same coin, each gets 50% of the reward as if they were the only agent collecting that coin. Only tested with OGCoin so far")
    parser.add_argument("--zero_vals", action="store_true", help="For testing/debug. Can also serve as another way to do no_baseline. Set all values to be 0 in Loaded Dice Calculation")
    parser.add_argument("--gae_lambda", type=float, default=1,
                        help="lambda for GAE (1 = monte carlo style, 0 = TD style)")
    parser.add_argument("--val_update_after_loop", action="store_true", help="Update values only after outer POLA loop finishes, not during the POLA loop")
    parser.add_argument("--std", type=float, default=0.1, help="standard deviation for initialization of policy/value parameters")
    parser.add_argument("--old_kl_div", action="store_true", help="Use the old version of KL div relative to just one batch of states at the beginning")
    # parser.add_argument("--layer_before_gru", action="store_true", help="Have a linear layer with ReLU before GRU")
    parser.add_argument("--inspect_ipd", action="store_true", help="Detailed (2 steps + start state) policy information in the IPD with full history")
    parser.add_argument("--layers_before_gru", type=int, default=2, choices=[0, 1, 2], help="Number of linear layers (with ReLU activation) before GRU, supported up to 2 for now")


    args = parser.parse_args()

    np.random.seed(args.seed)
    # jnp.manual_seed(args.seed)
    # TODO all the jax rng...


    if args.env == 'coin':
        assert args.grid_size == 3  # rest not implemented yet
        input_size = args.grid_size ** 2 * 4
        action_size = 4
        env = CoinGame()
    elif args.env == 'ipd':
        input_size = 6 # 3 * n_agents
        action_size = 2
        env = IPD(init_state_coop=args.init_state_coop)
    else:
        raise NotImplementedError("unknown env")
    vec_env_reset = jax.vmap(env.reset)
    vec_env_step = jax.vmap(env.step)



    key = jax.random.PRNGKey(args.seed)
    # key, subkey1, subkey2 = jax.random.split(key, 3)

    # if args.load_path is None:

    hidden_size = args.hidden_size

    key, key_p1, key_v1, key_p2, key_v2 = jax.random.split(key, 5)

    theta_p1 = RNN(num_outputs=action_size,
                       num_hidden_units=hidden_size)
    theta_v1 = RNN(num_outputs=1, num_hidden_units=hidden_size)

    theta_p1_params = theta_p1.init(key_p1, jnp.ones(
        [args.batch_size, input_size]), jnp.zeros(hidden_size))
    theta_v1_params = theta_v1.init(key_v1, jnp.ones(
        [args.batch_size, input_size]), jnp.zeros(hidden_size))

    theta_p2 = RNN(num_outputs=action_size,
                   num_hidden_units=hidden_size)
    theta_v2 = RNN(num_outputs=1, num_hidden_units=hidden_size)

    theta_p2_params = theta_p2.init(key_p2, jnp.ones(
        [args.batch_size, input_size]), jnp.zeros(hidden_size))
    theta_v2_params = theta_v2.init(key_v2, jnp.ones(
        [args.batch_size, input_size]), jnp.zeros(hidden_size))

    if args.optim.lower() == 'adam':
        theta_optimizer = optax.adam(learning_rate=args.lr_out)
        value_optimizer = optax.adam(learning_rate=args.lr_v)
    elif args.optim.lower() == 'sgd':
        theta_optimizer = optax.sgd(learning_rate=args.lr_out)
        value_optimizer = optax.sgd(learning_rate=args.lr_v)
    else:
        raise Exception("Unknown or Not Implemented Optimizer")

    trainstate_th1 = TrainState.create(apply_fn=theta_p1.apply,
                                           params=theta_p1_params,
                                           tx=theta_optimizer)
    trainstate_val1 = TrainState.create(apply_fn=theta_v1.apply,
                                            params=theta_v1_params,
                                            tx=value_optimizer)
    trainstate_th2 = TrainState.create(apply_fn=theta_p2.apply,
                                       params=theta_p2_params,
                                       tx=theta_optimizer)
    trainstate_val2 = TrainState.create(apply_fn=theta_v2.apply,
                                        params=theta_v2_params,
                                        tx=value_optimizer)
    # agent1 = Agent(subkey1, input_size, args.hidden_size, action_size, lr_p=args.lr_out, lr_v = args.lr_v)
    # agent2 = Agent(subkey2, input_size, args.hidden_size, action_size, lr_p=args.lr_out, lr_v = args.lr_v)
    if args.load_dir is not None:
        score_record = []
        vs_fixed_strats_score_record = [[], []]
        same_colour_coins_record = []
        diff_colour_coins_record = []
        coins_collected_info = (
        same_colour_coins_record, diff_colour_coins_record)

        assert args.load_prefix is not None
        restored_tuple = checkpoints.restore_checkpoint(ckpt_dir=args.load_dir,
                                                        target=(trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2,
                                                                coins_collected_info,
                                                                score_record,
                                                                vs_fixed_strats_score_record),
                                                        prefix=args.load_prefix)
        # agent1, agent2, coins_collected_info, prev_scores, vs_fixed_strat_scores = load_from_checkpoint()
        # print(jnp.stack(prev_scores))
        trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2 = restored_tuple


    use_baseline = True
    if args.no_baseline:
        use_baseline = False

    assert args.inner_steps >= 1
    # Use 0 lr if you want no inner steps... TODO fix this?
    assert args.outer_steps >= 1


    # # TODO REMOVE LATER TESTING ONLY
    #
    # key, subkey = jax.random.split(jax.random.PRNGKey(0))
    # score1, score2, rr_matches_amount, rb_matches_amount, br_matches_amount, bb_matches_amount, score1rec, score2rec = \
    #     eval_progress(key, trainstate_th1, trainstate_val1, trainstate_th2,
    #                   trainstate_val2)
    #
    # now = datetime.datetime.now()
    #
    # score_record = []
    # vs_fixed_strats_score_record = [[], []]
    # same_colour_coins_record = []
    # diff_colour_coins_record = []
    # coins_collected_info = (
    #     same_colour_coins_record, diff_colour_coins_record)
    #
    # if args.env == "coin":
    #     same_colour_coins = rr_matches_amount + bb_matches_amount
    #     diff_colour_coins = rb_matches_amount + br_matches_amount
    #     same_colour_coins_record.append(same_colour_coins)
    #     diff_colour_coins_record.append(diff_colour_coins)
    #
    # vs_fixed_strats_score_record[0].append(score1rec)
    # vs_fixed_strats_score_record[1].append(score2rec)
    #
    # score_record.append(jnp.stack((score1, score2)))
    #
    # print(coins_collected_info)
    # print(score_record)
    # print(vs_fixed_strats_score_record)
    #
    # checkpoints.save_checkpoint(ckpt_dir=args.save_dir,
    #                             target=(trainstate_th1, trainstate_val1,
    #                                     trainstate_th2, trainstate_val2,
    #                                     coins_collected_info,
    #                                     score_record,
    #                                     vs_fixed_strats_score_record),
    #                             step=1,
    #                             prefix=f"checkpoint_{now.strftime('%Y-%m-%d_%H-%M')}_seed{args.seed}_epoch")
    #
    # 1/0

    joint_scores = play(key, trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2,
                        args.opp_model)
