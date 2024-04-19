import jax.numpy as jnp
import jax.random

MIN_TEMPTATION_BONUS = 0
MAX_TEMPTATION_BONUS = 9

class IPD:
    """
    A two-agent vectorized environment.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    def __init__(self, init_state_coop=False, contrib_factor=1.33):
        cc = contrib_factor - 1.
        dd = 0.
        dc = contrib_factor / 2. # I defect when opp coop
        cd = contrib_factor / 2. - 1 # I coop when opp defect
        self.payout_mat = jnp.array([[dd, dc],[cd, cc]])
        # One hot state representation because this would scale to n agents
        self.almost_states = jnp.array(
            [[[1, 0, 0, 1, 0, 0], #DD (WE ARE BACK TO THE REPR OF FIRST AGENT, SECOND AGENT)
              [1, 0, 0, 0, 1, 0]], #DC
             [[0, 1, 0, 1, 0, 0], #CD
              [0, 1, 0, 0, 1, 0]]] #CC
        )
        if init_state_coop:
            self.init_state = jnp.array([0, 0, 1, 0, 0, 1, 0])
        else:
            self.init_state = jnp.array([0, 0, 0, 1, 0, 0, 1])

    def reset(self, unused_key):
        return self.init_state, self.init_state

    def step(self, state, ac0, ac1, rng_key):
        temptation_bonus = state[0]
        # assert MIN_TEMPTATION_BONUS <= temptation_bonus <= MAX_TEMPTATION_BONUS

        r0 = self.payout_mat[ac0, ac1] + jax.lax.select(ac0 < ac1, temptation_bonus, 0)
        r1 = self.payout_mat[ac1, ac0] + jax.lax.select(ac1 < ac0, temptation_bonus, 0)
        reward = (r0, r1)

        next_temptation_bonus = jax.random.randint(rng_key, (1,), MIN_TEMPTATION_BONUS,
                                                   MAX_TEMPTATION_BONUS, dtype=int)
        next_state = jnp.concatenate((next_temptation_bonus, self.almost_states[ac0, ac1]))

        return next_state, next_state, reward, None
