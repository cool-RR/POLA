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

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


from coin_game_jax import CoinGame



def reverse_cumsum(x, dim):
    return x + jnp.sum(x, dim=dim, keepdims=True) - jnp.cumsum(x, dim=dim)


device = 'cpu'

# DiCE operator
def magic_box(x):
    return jnp.exp(x - jax.lax.stop_gradient(x))


def dice_objective(self_logprobs, other_logprobs, rewards, values, end_state_v):
    # print(self_logprobs)
    # self_logprobs = jnp.stack(self_logprobs, dim=1)
    # other_logprobs = jnp.stack(other_logprobs, dim=1)

    # rewards = jnp.stack(rewards, dim=1)

    # print(rewards)
    # print(rewards.shape)
    # print(rewards.size)


    rewards = rewards.squeeze(-1)

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
        print("NOT YET FULLY CHECKED")
        raise NotImplementedError
        next_val_history = jnp.zeros((args.batch_size, args.rollout_len))
        next_val_history[:, :args.rollout_len - 1] = values[:, 1:args.rollout_len]
        next_val_history[:, -1] = end_state_v

        if args.zero_vals:
            next_val_history = jnp.zeros_like(next_val_history)
            values = jnp.zeros_like(values)

        advantages = jnp.zeros_like(values)
        lambd = args.gae_lambda # 1 here is essentially monte carlo (but with extrapolation of value in the end state)
        deltas = rewards + args.gamma * next_val_history.detach() - values.detach()
        gae = jnp.zeros_like(deltas[:, 0]).float()
        for i in range(deltas.size(1) - 1, -1, -1):
            gae = gae * args.gamma * lambd + deltas[:, i]
            advantages[:, i] = gae

        discounts = jnp.cumprod(
            args.gamma * jnp.ones((args.rollout_len), device=device),
            dim=0) / args.gamma

        discounted_advantages = advantages * discounts

        deps_up_to_t = (jnp.cumsum(stochastic_nodes, dim=1))

        deps_less_than_t = deps_up_to_t - stochastic_nodes  # take out the dependency in the given time step

        # Look at Loaded DiCE and GAE papers to see where this formulation comes from
        loaded_dice_rewards = ((magic_box(deps_up_to_t) - magic_box(
            deps_less_than_t)) * discounted_advantages).sum(dim=1).mean(dim=0)

        dice_objective = loaded_dice_rewards

    else:
        # dice objective:
        dice_objective = jnp.mean(
            jnp.sum(magic_box(dependencies) * discounted_rewards, axis=1))


    return -dice_objective  # want to minimize -objective



def value_loss(values, rewards, final_state_vals):
    # Fixed original value update which I'm almost certain is wrong

    discounts = jnp.cumprod(
        args.gamma * jnp.ones((args.rollout_len), device=device),
        dim=0) / args.gamma

    gamma_t_r_ts = rewards * discounts
    G_ts = reverse_cumsum(gamma_t_r_ts, dim=1)
    R_ts = G_ts / discounts

    final_val_discounted_to_curr = (args.gamma * discounts.flip(dims=[0])).expand((final_state_vals.shape[0], discounts.shape[0])) \
                                   * final_state_vals.expand((discounts.shape[0], final_state_vals.shape[0])).t()

    # You DO need a detach on these. Because it's the target - it should be detached. It's a target value.
    # Essentially a Monte Carlo style type return for R_t, except for the final state we also use the estimated final state value.
    # This becomes our target for the value function loss. So it's kind of a mix of Monte Carlo and bootstrap, but anyway you need the final value
    # because otherwise your value calculations will be inconsistent
    values_loss = (R_ts + final_val_discounted_to_curr - values) ** 2
    values_loss = values_loss.sum(dim=1).mean(dim=0)

    print("Values loss")
    print(values_loss)
    return values_loss


@jit
def act(key, env_batch_states, th_p_trainstate, th_p_trainstate_params, th_v_trainstate, th_v_trainstate_params, h_p, h_v, ret_logits=False):

    # TODO vectorize the env batch states, follow Chris code as example
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

    if ret_logits:
        return actions, log_probs_actions, ret_vals, h_p, h_v, categorical_act_probs, logits
    return actions, log_probs_actions, ret_vals, h_p, h_v, categorical_act_probs


def get_gradient(objective, theta):
    # create differentiable gradient for 2nd orders:
    grad_objective = jnp.autograd.grad(objective, (theta), create_graph=True)
    return grad_objective


# def step(theta1, theta2, values1, values2):
#     # just to evaluate progress:
#     env_state, obsv = env.reset()
#     s1 = obsv
#     s2 = obsv
#
#     score1 = 0
#     score2 = 0
#     h_p1, h_v1, h_p2, h_v2 = (
#     jnp.zeros(args.batch_size, args.hidden_size).to(device),
#     jnp.zeros(args.batch_size, args.hidden_size).to(device),
#     jnp.zeros(args.batch_size, args.hidden_size).to(device),
#     jnp.zeros(args.batch_size, args.hidden_size).to(device))
#     if args.env == "coin" or args.env == "ogcoin":
#         rr_matches_record, rb_matches_record, br_matches_record, bb_matches_record = 0., 0., 0., 0.
#
#     for t in range(args.rollout_len):
#         a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, theta1, values1, h_p1, h_v1)
#         a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(s2, theta2, values2, h_p2, h_v2)
#         (s1, s2), (r1, r2), _, info = env.step((a1, a2))
#         # cumulate scores
#         score1 += jnp.mean(r1) / float(args.rollout_len)
#         score2 += jnp.mean(r2) / float(args.rollout_len)
#
#         if args.env == "coin" or args.env == "ogcoin":
#             rr_matches, rb_matches, br_matches, bb_matches = info
#             rr_matches_record += rr_matches
#             rb_matches_record += rb_matches
#             br_matches_record += br_matches
#             bb_matches_record += bb_matches
#
#     if args.env == "coin" or args.env == "ogcoin":
#         return (score1, score2), (rr_matches_record, rb_matches_record, br_matches_record, bb_matches_record)
#
#     return (score1, score2), None


class RNN(nn.Module):
    num_outputs: int
    num_hidden_units: int

    def setup(self):
        self.GRUCell = nn.GRUCell()
        self.linear = nn.Dense(features=self.num_outputs)

    def __call__(self, x, carry):
        carry, x = self.GRUCell(carry, x)
        outputs = self.linear(x)
        return carry, outputs

    # def initialize_carry(self):
    #     return self.GRUCell.initialize_carry(
    #         jax.random.PRNGKey(0), (), self.num_hidden_units
    #     )

# def jnp.zeros(num_hidden_units):
#     # initializes as all 0s without regard to the prngkey
#     return nn.GRUCell.initialize_carry(
#         jax.random.PRNGKey(0), (), num_hidden_units
#     )



def get_policies_for_states(self):
    h_p1, h_v1 = (
        jnp.zeros(args.batch_size, args.hidden_size).to(device),
        jnp.zeros(args.batch_size, args.hidden_size).to(device))

    cat_act_probs = []

    for t in range(args.rollout_len):
        s1 = self.state_history[t]
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, self.theta_p,
                                                      self.theta_v, h_p1,
                                                      h_v1)
        cat_act_probs.append(cat_act_probs1)

    return jnp.stack(cat_act_probs, dim=1)

def get_other_policies_for_states(self, other_theta, other_values, state_history):
    # Perhaps really should not be named 1
    # Well this also doesn't even have to be other, this works fine for any theta and vals as long as the state history is correct (corresponds to the theta and values you are using)
    h_p1, h_v1 = (
        jnp.zeros(args.batch_size, args.hidden_size).to(device),
        jnp.zeros(args.batch_size, args.hidden_size).to(device))

    cat_act_probs = []

    for t in range(args.rollout_len):
        s1 = state_history[t]
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(s1, other_theta,
                                                      other_values, h_p1,
                                                      h_v1)
        cat_act_probs.append(cat_act_probs1)

    return jnp.stack(cat_act_probs, dim=1)

def get_other_logits_values_for_states(self, other_theta, other_values, state_history):
    # Same comments as above. Questionable variable naming here
    h_p1, h_v1 = (
        jnp.zeros(args.batch_size, args.hidden_size).to(device),
        jnp.zeros(args.batch_size, args.hidden_size).to(device))

    logits_hist = []
    vals_hist = []


    for t in range(args.rollout_len):
        s1 = state_history[t]
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1, logits = act(s1, other_theta,
                                                      other_values, h_p1,
                                                      h_v1, ret_logits=True)
        logits_hist.append(logits)
        vals_hist.append(v1)

    final_state = state_history[-1]
    # act just to get the final state values
    a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(final_state, other_theta,
                                                  other_values,
                                                  h_p1, h_v1)
    final_state_vals = v1

    if use_baseline:
        return jnp.stack(logits_hist, dim=1), jnp.stack(vals_hist, dim=1), final_state_vals
    else:
        return jnp.stack(logits_hist, dim=1), None, None

@jit
def env_step(stuff, unused):
    # TODO should make this agent agnostic? Or have a flip switch? Can reorganize later
    key, env_state, obs1, obs2, \
    trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params, \
    trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params, \
    h_p1, h_v1, h_p2, h_v2 = stuff
    key, sk1, sk2, skenv = jax.random.split(key, 4)
    a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(sk1, obs1,
                                                  trainstate_th1, trainstate_th1_params,
                                                  trainstate_val1, trainstate_val1_params,
                                                  h_p1, h_v1)
    a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(sk2, obs2,
                                                  trainstate_th2,
                                                  trainstate_th2_params,
                                                  trainstate_val2,
                                                  trainstate_val2_params,
                                                  h_p2,
                                                  h_v2)
    skenv = jax.random.split(skenv, args.batch_size)
    env_state, new_obs, (r1, r2) = vec_env_step(env_state, a1, a2, skenv)
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

    return stuff, (aux1, aux2)

@partial(jit, static_argnums=(9, 10))
def in_lookahead(key, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
                 trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params, first_inner_step=False,
                 other_agent=2):
    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    # env_state, obsv = env.reset(subkey)
    env_state, obsv = vec_env_reset(env_subkeys)

    obs1 = obsv
    obs2 = obsv

    # print(obsv.shape)

    # other_memory = Memory()
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

    if first_inner_step:
        cat_act_probs_other = []
        other_state_history = []
        if other_agent == 2:
            other_state_history.append(obs2)
        else:
            other_state_history.append(obs1)


    stuff = (key, env_state, obs1, obs2,
             trainstate_th1, trainstate_th1_params, trainstate_val1,
             trainstate_val1_params,
             trainstate_th2, trainstate_th2_params, trainstate_val2,
             trainstate_val2_params,
             h_p1, h_v1, h_p2, h_v2)

    stuff, aux = jax.lax.scan(env_step, stuff, None, args.rollout_len)
    aux1, aux2 = aux

    key, env_state, obs1, obs2, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,\
    trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params, h_p1, h_v1, h_p2, h_v2 = stuff

    key, subkey1, subkey2 = jax.random.split(key, 3)

    # TODO remove the redundancies
    if other_agent == 2:
        cat_act_probs2_list, obs2_list, lp2_list, lp1_list, v2_list, r2_list, a2_list, a1_list = aux2
        if first_inner_step:
            cat_act_probs_other.extend(cat_act_probs2_list)
            other_state_history.extend(obs2_list)

        a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(subkey2, obs2,
                                                      trainstate_th2,
                                                      trainstate_th2_params,
                                                      trainstate_val2,
                                                      trainstate_val2_params,
                                                      h_p2, h_v2)
        end_state_v2 = v2

        inner_agent_objective = dice_objective(self_logprobs=lp2_list,
                                               other_logprobs=lp1_list,
                                               rewards=r2_list,
                                               values=v2_list,
                                               end_state_v=end_state_v2)

        print(
            f"Inner Agent (Agent 2) episode return avg {r2_list.sum(axis=0).mean()}")


    else:
        assert other_agent == 1
        cat_act_probs1_list, obs1_list, lp1_list, lp2_list, v1_list, r1_list, a1_list, a2_list = aux1
        if first_inner_step:
            cat_act_probs_other.extend(cat_act_probs1_list)
            other_state_history.extend(obs1_list)

        a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(subkey1, obs1,
                                                      trainstate_th1,
                                                      trainstate_th1_params,
                                                      trainstate_val1,
                                                      trainstate_val1_params,
                                                      h_p1, h_v1)
        end_state_v1 = v1

        inner_agent_objective = dice_objective(self_logprobs=lp1_list,
                                               other_logprobs=lp2_list,
                                               rewards=r1_list,
                                               values=v1_list,
                                               end_state_v=end_state_v1)

        print(
            f"Inner Agent (Agent 1) episode return avg {r1_list.sum(axis=0).mean()}")

    # print(r2_list)
    # print(r2_list.shape)
    # print(other_state_history)

    # act just to get the final state values

    # print(obs2)
    # print(obs2_list[-1] - obs2)

    # TODO remove this end state whatever nonsense, and have to update all the calculations which use the list of values
    # so that now the end state value is just the -1 element of the list
    # and the rest of the value calculations are all up to but excluding the last element

    if not first_inner_step:
        print("KL div calc not yet done")
        # TODO: edit the below. Use scan everywhere instead of for loops.
        # curr_pol_probs = self.get_other_policies_for_states(other_th_trainstate, other_val_trainstate, self.other_state_history)
        # kl_div = jnp.nn.functional.kl_div(jnp.log(curr_pol_probs), self.ref_cat_act_probs_other.detach(), log_target=False, reduction='batchmean')
        # print(kl_div)

        # other_objective += args.inner_beta * kl_div # we want to min kl div

    if first_inner_step:
        print("KL div calc not yet done")
        # TODO CHECK ALL OF THIS STUFF
        # use as ref for KL div calc
        ref_cat_act_probs_other = cat_act_probs_other
        other_state_history = other_state_history

    # print(f"Agent 2 reward sum {r2_list.sum(axis=0)}")

    # return grad
    return inner_agent_objective


def inner_step_get_grad_otheragent2(stuff, unused):
    key, trainstate_th1_, trainstate_val1_, trainstate_th2_, trainstate_val2_, first_inner_step = stuff
    key, subkey = jax.random.split(key)

    other_agent_obj_grad_fn = jax.grad(in_lookahead, argnums=[6, 8])

    grad_th, grad_v = other_agent_obj_grad_fn(subkey,
                                              trainstate_th1_,
                                              trainstate_th1_.params,
                                              trainstate_val1_,
                                              trainstate_val1_.params,
                                              trainstate_th2_,
                                              trainstate_th2_.params,
                                              trainstate_val2_,
                                              trainstate_val2_.params,
                                              first_inner_step=first_inner_step,
                                              other_agent=2)

    # update other's theta: NOTE HERE THIS IS JUST AN SGD UPDATE
    trainstate_th2_ = trainstate_th2_.apply_gradients(grads=grad_th)

    # TODO when value update the inner model? Do it at all?
    if use_baseline:
        raise NotImplementedError
        # model_state_val_ = model_state_val_.apply_gradients(
        #     grads=grad_v)
        # return model_state_th_, model_state_val_

    # Since we only need the final trainstate, and not every trainstate every step of the way, no need for aux here
    stuff = (key, trainstate_th1_, trainstate_val1_, trainstate_th2_, trainstate_val2_, first_inner_step)
    aux = None

    return stuff, aux


def inner_step_get_grad_otheragent1(stuff, unused):
    key, trainstate_th1_, trainstate_val1_, trainstate_th2_, trainstate_val2_, first_inner_step = stuff
    key, subkey = jax.random.split(key)

    other_agent_obj_grad_fn = jax.grad(in_lookahead,
                                       argnums=[2, 4])

    grad_th, grad_v = other_agent_obj_grad_fn(subkey,
                                              trainstate_th1_,
                                              trainstate_th1_.params,
                                              trainstate_val1_,
                                              trainstate_val1_.params,
                                              trainstate_th2_,
                                              trainstate_th2_.params,
                                              trainstate_val2_,
                                              trainstate_val2_.params,
                                              first_inner_step=first_inner_step,
                                              other_agent=1)

    # update other's theta: NOTE HERE THIS IS JUST AN SGD UPDATE

    trainstate_th1_ = trainstate_th1_.apply_gradients(grads=grad_th)

    # TODO when value update the inner model? Do it at all?
    if use_baseline:
        raise NotImplementedError
        # model_state_val_ = model_state_val_.apply_gradients(
        #     grads=grad_v)
        # return model_state_th_, model_state_val_

    # Since we only need the final trainstate, and not every trainstate every step of the way, no need for aux here
    stuff = (key, trainstate_th1_, trainstate_val1_, trainstate_th2_, trainstate_val2_, first_inner_step)
    aux = None

    return stuff, aux


# @partial(jit, static_argnums=(10))
# TODO can replace other_agent with each of the theta_p or theta_v or whatever. This could be one way to remove the agent class
# So that things can be jittable. Everything has to be pure somehow
def inner_steps_plus_update(key, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
                 trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params, n_lookaheads, other_agent=2 ):
    # TODO ensure that this starts from scratch (well the agent 2 policy before the updates happen)
    trainstate_th1_ = TrainState.create(apply_fn=trainstate_th1.apply_fn,
                                        params=trainstate_th1_params,
                                        tx=optax.sgd(
                                            learning_rate=args.lr_in))
    trainstate_val1_ = TrainState.create(apply_fn=trainstate_val1.apply_fn,
                                         params=trainstate_val1_params,
                                         tx=optax.sgd(
                                             learning_rate=args.lr_in))

    trainstate_th2_ = TrainState.create(apply_fn=trainstate_th2.apply_fn,
                                         params=trainstate_th2_params,
                                         tx=optax.sgd(
                                             learning_rate=args.lr_in))
    trainstate_val2_ = TrainState.create(apply_fn=trainstate_val2.apply_fn,
                                          params=trainstate_val2_params,
                                          tx=optax.sgd(
                                              learning_rate=args.lr_in))


    key, subkey = jax.random.split(key)

    # if other_agent == 2:
    #     other_agent_obj_grad_fn = jax.grad(in_lookahead,
    #                                        argnums=[6, 8])
    # else:
    #     assert other_agent == 1
    #     other_agent_obj_grad_fn = jax.grad(in_lookahead,
    #                                        argnums=[2, 4])

    subkey, subkey1 = jax.random.split(subkey)
    stuff = (subkey1, trainstate_th1_, trainstate_val1_, trainstate_th2_,
             trainstate_val2_, True)
    if other_agent == 2:
        stuff, aux = inner_step_get_grad_otheragent2(stuff, None)
    else:
        assert other_agent == 1
        stuff, aux = inner_step_get_grad_otheragent1(stuff, None)

    _, trainstate_th1_, _, trainstate_th2_, _, _ = stuff

    # if n_lookaheads > 1:
    #     stuff = (subkey2, trainstate_th1_, trainstate_val1_, trainstate_th2_,
    #              trainstate_val2_, False)
    #     if other_agent == 2:
    #         stuff, aux = jax.lax.scan(inner_step_get_grad_otheragent2, stuff, None, n_lookaheads-1)
    #     else:
    #         assert other_agent == 1
    #         stuff, aux = jax.lax.scan(inner_step_get_grad_otheragent1, stuff,
    #                                   None, n_lookaheads - 1)
    #
    #     _, trainstate_th1_, _, trainstate_th2_, _, _ = stuff

    # TODO replace this with a lax.scan too
    if n_lookaheads > 1:
        for inner_step in range(n_lookaheads-1):
            subkey, subkey1 = jax.random.split(subkey)
            if other_agent == 2:
                stuff = (subkey1, trainstate_th1_, trainstate_val1_, trainstate_th2_,
                             trainstate_val2_, False)
                stuff, aux  = inner_step_get_grad_otheragent2(stuff, None)
            else:
                stuff = (subkey1, trainstate_th1_, trainstate_val1_, trainstate_th2_,
                trainstate_val2_, False)
                stuff, aux  = inner_step_get_grad_otheragent1(stuff, None)
            _, trainstate_th1_, _, trainstate_th2_, _, _ = stuff


    if other_agent == 2:
        return trainstate_th2_, None
    else:
        return trainstate_th1_, None

    # return model_state_th_, None

def out_lookahead(key, trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
                  trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,
                  first_outer_step=False, agent_copy_for_val_update=None, self_agent=1):
    # AGENT COPY IS A COPY OF SELF, NOT OF OTHER. Used so that you can update the value function
    # while POLA is running in the outer loop, but the outer loop steps still calculate the loss
    # for the policy based on the old, static value function (this is for the val_update_after_loop stuff).
    # This should hopefully help with issues that might arise if value function is changing during the POLA update

    keys = jax.random.split(key, args.batch_size + 1)
    key, env_subkeys = keys[0], keys[1:]

    # env_state, obsv = env.reset(subkey)
    env_state, obsv = vec_env_reset(env_subkeys)

    obs1 = obsv
    obs2 = obsv

    # print(obsv.shape)

    # other_memory = Memory()
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

    state_history_for_vals = []
    state_history_for_vals.append(obs1)
    rew_history_for_vals = []

    if first_outer_step:
        cat_act_probs_self = []
        state_history_for_kl_div = []
        state_history_for_kl_div.append(obs1)


    stuff = (
    key, env_state, obs1, obs2,
    trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,
    trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,
    h_p1, h_v1, h_p2, h_v2)

    stuff, aux = jax.lax.scan(env_step, stuff, None, args.rollout_len)
    aux1, aux2 = aux

    key, env_state, obs1, obs2, \
    trainstate_th1, trainstate_th1_params, trainstate_val1, trainstate_val1_params,\
    trainstate_th2, trainstate_th2_params, trainstate_val2, trainstate_val2_params,\
    h_p1, h_v1, h_p2, h_v2 = stuff

    if self_agent == 1:
        cat_act_probs1_list, obs1_list, lp1_list, lp2_list, v1_list, r1_list, a1_list, a2_list = aux1
        if first_outer_step:
            cat_act_probs_self.extend(cat_act_probs1_list)
            state_history_for_kl_div.extend(obs1_list)

        key, subkey = jax.random.split(key)
        # act just to get the final state values
        a1, lp1, v1, h_p1, h_v1, cat_act_probs1 = act(subkey, obs1,
                                                      trainstate_th1,
                                                      trainstate_th1_params,
                                                      trainstate_val1,
                                                      trainstate_val1_params,
                                                      h_p1, h_v1)

        end_state_v = v1
        objective = dice_objective(self_logprobs=lp1_list,
                                   other_logprobs=lp2_list,
                                   rewards=r1_list, values=v1_list,
                                   end_state_v=end_state_v)
        print(f"Agent 1 episode return avg {r1_list.sum(axis=0).mean()}")
    else:
        assert self_agent == 2
        cat_act_probs2_list, obs2_list, lp2_list, lp1_list, v2_list, r2_list, a2_list, a1_list = aux2
        if first_outer_step:
            cat_act_probs_self.extend(cat_act_probs2_list)
            state_history_for_kl_div.extend(obs2_list)

        key, subkey = jax.random.split(key)
        # act just to get the final state values
        a2, lp2, v2, h_p2, h_v2, cat_act_probs2 = act(subkey, obs2,
                                                      trainstate_th2,
                                                      trainstate_th2_params,
                                                      trainstate_val2,
                                                      trainstate_val2_params,
                                                      h_p2, h_v2)
        end_state_v = v2
        objective = dice_objective(self_logprobs=lp2_list,
                                   other_logprobs=lp1_list,
                                   rewards=r2_list, values=v2_list,
                                   end_state_v=end_state_v)
        print(f"Agent 2 episode return avg {r2_list.sum(axis=0).mean()}")

    if not first_outer_step:
        print("KL div calc not yet done")
        # TODO: edit the below. Use scan everywhere instead of for loops.
        # curr_pol_probs = self.get_other_policies_for_states(other_th_trainstate, other_val_trainstate, self.other_state_history)
        # kl_div = jnp.nn.functional.kl_div(jnp.log(curr_pol_probs), self.ref_cat_act_probs_other.detach(), log_target=False, reduction='batchmean')
        # print(kl_div)
        # curr_pol_probs = self.get_policies_for_states()
        # kl_div = jnp.nn.functional.kl_div(jnp.log(curr_pol_probs),
        #                                   self.ref_cat_act_probs.detach(),
        #                                   log_target=False,
        #                                   reduction='batchmean')
        # print(kl_div)

        # other_objective += args.inner_beta * kl_div # we want to min kl div

    if first_outer_step:
        # TODO check this is right (and fits with agent 1 vs agent 2)
        # use as ref for KL div calc
        ref_cat_act_probs = cat_act_probs_self
        state_history = state_history_for_kl_div

    # return grad
    return objective


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

# @jit
def play(key, trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2,
         # theta_v1, theta_v1_params, agent1_value_optimizer,
         # theta_p2, theta_p2_params, agent2_theta_optimizer,
         # theta_v2, theta_v2_params, agent2_value_optimizer,
         n_lookaheads, outer_steps, use_opp_model=False): #,prev_scores=None, prev_coins_collected_info=None):
    joint_scores = []
    score_record = []
    # You could do something like the below and then modify the code to just be one continuous record that includes past values when loading from checkpoint
    # if prev_scores is not None:
    #     score_record = prev_scores
    # I'm tired though.
    vs_fixed_strats_score_record = [[], []]

    print("start iterations with", n_lookaheads, "inner steps and", outer_steps, "outer steps:")
    same_colour_coins_record = []
    diff_colour_coins_record = []

    agent2_theta_p_model, agent1_theta_p_model = None, None
    agent2_theta_v_model, agent1_theta_v_model = None, None

    for update in range(args.n_update):
        # TODO RECREATE TRAINSTATES IF NEEDED
        model_state_th1 = TrainState.create(apply_fn=trainstate_th1.apply_fn,
                                                    params=trainstate_th1.params,
                                                    tx=trainstate_th1.tx)
        model_state_val1 = TrainState.create(apply_fn=trainstate_val1.apply_fn,
                                            params=trainstate_val1.params,
                                            tx=trainstate_val1.tx)
        model_state_th2 = TrainState.create(apply_fn=trainstate_th2.apply_fn,
                                                    params=trainstate_th2.params,
                                                    tx=trainstate_th2.tx)
        model_state_val2 = TrainState.create(apply_fn=trainstate_val2.apply_fn,
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

        for outer_step in range(outer_steps):

            # WRAP FROM HERE

            key, subkey = jax.random.split(key)
            model_state_th2_, model_state_val2_ = \
                inner_steps_plus_update(subkey,
                                        trainstate_th1, trainstate_th1.params, trainstate_val1, trainstate_val1.params,
                                        trainstate_th2, trainstate_th2.params, trainstate_val2, trainstate_val2.params,
                                        n_lookaheads, other_agent=2)

            # TODO perhaps one way to get around this conditional outer step with kind of global storage
            # is that we should have one call to the lookaheads in the first time step
            # and then a scan over the rest of the x steps

            # update own parameters from out_lookahead:
            if outer_step == 0:
                first_outer_step = True
            else:
                first_outer_step = False

            if use_baseline:
                objective = out_lookahead(key, model_state_th1, model_state_th1.params, model_state_val1, model_state_val1.params,
                                          model_state_th2_, model_state_th2_.params, model_state_val2_, model_state_val2_.params,
                                     first_outer_step=first_outer_step, agent_copy_for_val_update=agent1_copy_for_val_update, self_agent=1)
            else:
                objective = out_lookahead(key, model_state_th1,
                                          model_state_th1.params,
                                          None,
                                          None,
                                          model_state_th2_, model_state_th2_.params, None, None,
                                          first_outer_step=first_outer_step,
                                          agent_copy_for_val_update=agent1_copy_for_val_update,
                                          self_agent=1)

            # TO HERE in a function, maybe outer_loop_step or something, or same way I named the inner_steps_plus_update
            # To get the grad on the outer loop
            # TODO then check that the grad is correct, with proper higher order gradients (it should be)
            # And then start running
            # Oh but we have to JIT everything first
            # And we also need to make sure that it all works correctly for the second agent too
            # Be very careful checking for bugs
            # I should probably re-go-over everything in the code once all done.

            # TODO Aug 18 Now we need the grad from the objective
            # TO do this, we can wrap this whole block (everything after (for outer_step in ...))
            # in another function. Then this function can take in the params of agent 1
            # Right so maybe get rid of the agent stuff, just directly use the params/trainstate of agent 1
            # instead of accessing the agent parameters
            # Store everything directly in trainstates, try to get rid of all the agent stuff as well
            # Define just as a function
            # Maybe even just define everything else as a function too, without calls to agent
            # ideally there should be 0 calls to agent.anything.
            # Then once I have this wrapper function I can take the grad again.

            1/0


        # TODO essentially what I think I need is to have the out lookahead include the in lookahead component... or a new outer step function
        # kind of like what I have in the other file, where then I can use that outer step function to take the grad of agent 1 params after
        # agent 2 has done a bunch of inner steps.

        for outer_step in range(outer_steps):
            th1_to_copy = start_theta1
            val1_to_copy = start_val1
            if use_opp_model:
                th1_to_copy = agent1_theta_p_model
                val1_to_copy = agent1_theta_v_model

            theta1_ = [tp.detach().clone().requires_grad_(True) for tp in
                       th1_to_copy]
            values1_ = [tv.detach().clone().requires_grad_(True) for tv in
                        val1_to_copy]

            for inner_step in range(n_lookaheads):
                # estimate other's gradients from in_lookahead:
                if inner_step == 0:
                    grad1 = agent2.in_lookahead(theta1_, values1_, first_inner_step=True)
                else:
                    grad1 = agent2.in_lookahead(theta1_, values1_, first_inner_step=False)
                # update other's theta
                theta1_ = [theta1_[i] - args.lr_in * grad1[i] for i in
                           range(len(theta1_))]

            if outer_step == 0:
                agent2.out_lookahead(theta1_, values1_, first_outer_step=True, agent_copy_for_val_update=agent2_copy_for_val_update)
            else:
                agent2.out_lookahead(theta1_, values1_, first_outer_step=False, agent_copy_for_val_update=agent2_copy_for_val_update)


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
        score, info = step(agent1.theta_p, agent2.theta_p, agent1.theta_v,
                           agent2.theta_v)

        print("Eval vs Fixed strategies not yet implemented")
        # print("Eval vs Fixed Strategies:")
        # score1rec = []
        # score2rec = []
        # for strat in ["alld", "allc", "tft"]:
        #     print(f"Playing against strategy: {strat.upper()}")
        #     score1, _ = eval_vs_fixed_strategy(agent1.theta_p, agent1.theta_v, strat, i_am_red_agent=True)
        #     score1rec.append(score1[0])
        #     print(f"Agent 1 score: {score1[0]}")
        #     score2, _ = eval_vs_fixed_strategy(agent2.theta_p, agent2.theta_v, strat, i_am_red_agent=False)
        #     score2rec.append(score2[1])
        #     print(f"Agent 2 score: {score2[1]}")
        #
        #     print(score1)
        #     print(score2)
        #
        # score1rec = jnp.stack(score1rec)
        # score2rec = jnp.stack(score2rec)
        # vs_fixed_strats_score_record[0].append(score1rec)
        # vs_fixed_strats_score_record[1].append(score2rec)

        rr_matches, rb_matches, br_matches, bb_matches = info
        same_colour_coins = (rr_matches + bb_matches).item()
        diff_colour_coins = (rb_matches + br_matches).item()
        same_colour_coins_record.append(same_colour_coins)
        diff_colour_coins_record.append(diff_colour_coins)

        joint_scores.append(0.5 * (score[0] + score[1]))
        score = jnp.stack(score)
        score_record.append(score)

        # print
        if update % args.print_every == 0:
            print("*" * 10)
            print("Epoch: {}".format(update + 1), flush=True)
            print(f"Score 0: {score[0]}")
            print(f"Score 1: {score[1]}")
            if args.env == "coin" or args.env == "ogcoin":
                print("Same coins: {}".format(same_colour_coins))
                print("Diff coins: {}".format(diff_colour_coins))
                print("RR coins {}".format(rr_matches))
                print("RB coins {}".format(rb_matches))
                print("BR coins {}".format(br_matches))
                print("BB coins {}".format(bb_matches))


    return joint_scores




if __name__ == "__main__":
    parser = argparse.ArgumentParser("NPLOLA")
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
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="Epochs between checkpoint save")
    parser.add_argument("--load_path", type=str, default=None, help="Give path if loading from a checkpoint")
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
    parser.add_argument("--env", type=str, default="ogcoin",
                        choices=["ipd", "ogcoin"])
    parser.add_argument("--hist_one", action="store_true", help="Use one step history (no gru or rnn, just one step history)")
    parser.add_argument("--print_info_each_outer_step", action="store_true", help="For debugging/curiosity sake")
    parser.add_argument("--init_state_coop", action="store_true", help="For IPD only: have the first state be CC instead of a separate start state")
    parser.add_argument("--split_coins", action="store_true", help="If true, then when both agents step on same coin, each gets 50% of the reward as if they were the only agent collecting that coin. Only tested with OGCoin so far")
    parser.add_argument("--zero_vals", action="store_true", help="For testing/debug. Can also serve as another way to do no_baseline. Set all values to be 0 in Loaded Dice Calculation")
    parser.add_argument("--gae_lambda", type=float, default=1,
                        help="lambda for GAE (1 = monte carlo style, 0 = TD style)")
    parser.add_argument("--val_update_after_loop", action="store_true", help="Update values only after outer POLA loop finishes, not during the POLA loop")
    parser.add_argument("--std", type=float, default=0.1, help="standard deviation for initialization of policy/value parameters")


    args = parser.parse_args()

    np.random.seed(args.seed)
    # jnp.manual_seed(args.seed)
    # TODO all the jax rng...

    assert args.grid_size == 3 # rest not implemented yet
    input_size = args.grid_size ** 2 * 4
    action_size = 4
    env = CoinGame()
    vec_env_reset = jax.vmap(env.reset)
    vec_env_step = jax.vmap(env.step)



    key = jax.random.PRNGKey(args.seed)
    # key, subkey1, subkey2 = jax.random.split(key, 3)

    if args.load_path is None:

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
    else:
        raise NotImplementedError("checkpointing not done yet")
        # agent1, agent2, coins_collected_info, prev_scores, vs_fixed_strat_scores = load_from_checkpoint()
        # print(jnp.stack(prev_scores))

    use_baseline = True
    if args.no_baseline:
        use_baseline = False



    joint_scores = play(key, trainstate_th1, trainstate_val1, trainstate_th2, trainstate_val2,
                        args.inner_steps, args.outer_steps, args.opp_model)
