"""Pinned Kinetix car evaluator, loaded only by car evaluation commands."""
from importlib.metadata import version
import jax
import jax.numpy as jnp
import numpy as np
from kinetix.environment import make_kinetix_env, ActionType, ObservationType
from kinetix.util.saving import load_from_json_file

KINETIX_COMMIT = '80ee9c292837c825cafaf9889323fdcc212efd1b'


class CarEvaluator:
    def __init__(self):
        level, static, self.params = load_from_json_file('l/h2_car_ramp.json')
        self.params = self.params.replace(dense_reward_scale=0.2)
        self.env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT,
                                    lambda rng: level, self.params, static, auto_reset=False)
        obs, _ = self.env.reset(jax.random.PRNGKey(0), self.params)
        if obs.shape != (679,):
            raise ValueError('Kinetix car observation must have 679 entries')
        self.simulate = jax.jit(jax.vmap(self.episode, in_axes=(None, 0)))
        self.metadata = dict(kinetix_commit=KINETIX_COMMIT, jax=jax.__version__,
                             kinetix=version('kinetix-env'), max_steps=256, dtype='float32')

    def episode(self, weights, key):
        key, reset = jax.random.split(key)
        obs, state = self.env.reset(reset, self.params)

        def step(carry, unused):
            obs, state, key, ended, succeeded, failed, length, total = carry
            key, sample, physics = jax.random.split(key, 3)
            hidden = jax.nn.relu(obs @ weights['embed.W'] + weights['embed.b'])
            for index in range(5):
                hidden = jnp.tanh(hidden @ weights[f'actor{index}.W'] + weights[f'actor{index}.b'])
            logits = hidden @ weights['actor_head.W'] + weights['actor_head.b']
            keys = jax.random.split(sample, 6)
            offsets = ((0,3),(3,6),(6,9),(9,12),(12,14),(14,16))
            action = jnp.stack([jax.random.categorical(k, logits[a:b]) for k,(a,b) in zip(keys, offsets)])
            next_obs, next_state, reward, done, info = self.env.step(physics, state, action, self.params)
            active = ~ended
            success = active & done & info['GoalR']
            red = active & done & ~info['GoalR'] & (reward < -0.5)
            return (next_obs, next_state, key, ended | done, succeeded | success, failed | red,
                    length + active, total + jnp.where(active, reward, 0)), None

        start = (obs, state, key, jnp.array(False), jnp.array(False), jnp.array(False), jnp.array(0), jnp.array(0.0))
        result = jax.lax.while_loop(lambda carry: (~carry[3]) & (carry[6] < 256),
                                    lambda carry: step(carry, None)[0], start)
        return dict(success=result[4], red=result[5], steps=result[6], returns=result[7])


    def evaluate(self, weights, episodes, seed, batch_size=32, progress=None):
        weights = {key: jnp.asarray(value, dtype=jnp.float32) for key, value in weights.items()}
        rows = []
        for offset in range(0, episodes, batch_size):
            count = min(batch_size, episodes-offset)
            keys = jnp.stack([jax.random.PRNGKey(seed+i) for i in range(offset, offset+count)])
            batch = jax.device_get(self.simulate(weights, keys))
            for i in range(count):
                rows.append(dict(seed=seed+offset+i, **{key: value[i].item() for key, value in batch.items()}))
            result = dict(successes=sum(row['success'] for row in rows),
                          red_collisions=sum(row['red'] for row in rows),
                          success_rate=float(np.mean([row['success'] for row in rows])),
                          mean_steps=float(np.mean([row['steps'] for row in rows])), episodes=list(rows))
            if progress:
                progress(result)
        return result

    def trajectory(self, weights, seed):
        weights = {key: jnp.asarray(value, dtype=jnp.float32) for key, value in weights.items()}
        key, reset = jax.random.split(jax.random.PRNGKey(seed))
        obs, state = self.env.reset(reset, self.params)

        @jax.jit
        def step(obs, state, key):
            key, sample, physics = jax.random.split(key, 3)
            hidden = jax.nn.relu(obs @ weights['embed.W'] + weights['embed.b'])
            for i in range(5):
                hidden = jnp.tanh(hidden @ weights[f'actor{i}.W'] + weights[f'actor{i}.b'])
            logits = hidden @ weights['actor_head.W'] + weights['actor_head.b']
            offsets = ((0,3),(3,6),(6,9),(9,12),(12,14),(14,16))
            action = jnp.stack([jax.random.categorical(k, logits[a:b]) for k, (a,b)
                                in zip(jax.random.split(sample,6), offsets)])
            next_obs, state, reward, done, info = self.env.step(physics, state, action, self.params)
            return next_obs, state, key, reward, done, info, action, jax.nn.softmax(logits[:3])

        rows = []
        for number in range(256):
            next_obs, state, key, reward, done, info, action, probabilities = step(obs, state, key)
            if number % 4 == 0 or bool(done):
                rows.append(dict(step=number, body=obs[52:56].tolist(), green=obs[234:238].tolist(),
                                 action=int(action[0]), probabilities=probabilities.tolist(),
                                 reward=float(reward), done=bool(done), success=bool(info['GoalR'])))
            obs = next_obs
            if bool(done):
                break
        return dict(seed=seed, fields=['x','y','vx','vy'], steps=number+1, trajectory=rows)
