#####

# Based on:
# https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/

#####

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# run the model to play one episode, and return the frames
def render_policy_net(model, n_max_steps=500, seed=42):
    frames = []
    env = gym.make("CartPole-v1")
    env.seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    for step in range(n_max_steps):
        frames.append(env.render(mode="rgb_array"))
        left_proba = model.predict(obs.reshape(1, -1))
        action = int(np.random.rand() > left_proba)
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    return frames


def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        # makes an action more likely
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    env.render()
    return obs, reward, done, grads


def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads


def discount_rewards(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]



def main():
    env = gym.make('CartPole-v1')
    env.seed(42)
    n_inputs = env.observation_space.shape[0]
    model = keras.models.Sequential([
        keras.layers.Dense(10, activation="relu", input_shape=[n_inputs]),
        keras.layers.Dense(5, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    n_iterations = 100
    n_episodes_per_update = 10
    n_max_steps = 500               # 500 is the max possible score
    discount_rate = 0.95
    optimizer = keras.optimizers.Adam(lr=0.01)
    loss_fn = keras.losses.binary_crossentropy


    for iteration in range(n_iterations):
        all_rewards, all_grads = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model, loss_fn)
        total_rewards = sum(map(sum, all_rewards))                    
        print("\rIteration: {}, mean rewards: {:.1f}".format(iteration, total_rewards / n_episodes_per_update), end="") 
        all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[episode_index][step][var_index]
                for episode_index, final_rewards in enumerate(all_final_rewards)
                    for step, final_reward in enumerate(final_rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

    env.close()


if __name__ == '__main__':
    main()