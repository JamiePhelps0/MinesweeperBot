import tensorflow as tf
import numpy as np
import pickle
import multiprocessing
from tqdm import tqdm
from trainEnvFC import TrainEnv

np.set_printoptions(linewidth=400)


"""
For the Minesweeper Game, instead of learning the expected discounted reward given a state action pair (Q(s, a)), it 
should be easier to directly learn the reward function r(s, a). Taking a greedy action in a minesweeper state does not
affect long term rewards. I refer to this as Deep Reward Learning (DRN). This is similar to DQN with gamma = 0.
Rewards from Minesweeper environment: 
1 if action does not cause episode termination OR all mines found, 
0 if action selects mine OR selects position that is already known (useless action, used instead of an action mask)
"""

batch_size = 512

size = [30, 16]

num_actions = size[0] * size[1]


def test(net, tests=100):
    """
    :param net: tf.keras.Model: network to be evaluated
    :param tests: int: number of episodes to run
    :return: float: average return of policy in env
    Runs all episodes in parallel
    """
    returns = []
    envs = [TrainEnv(size, 99) for _ in range(tests)]
    envs = mp_reset(envs)
    actions = 255 * np.ones((tests, ), dtype=np.int32)  # np.random.randint(0, num_actions, (tests,))
    envs, states, _, dones = mp_act(envs, actions)
    batch_rewards = sum(1 - np.array(dones).flatten())
    returns.append(batch_rewards)
    while not all(dones):
        actions = np.array(net(states))
        actions = np.argmax(actions, axis=-1)
        envs, states, _, dones = mp_act(envs, actions)
        batch_rewards = sum(1 - np.array(dones).flatten())
        returns.append(batch_rewards)
        envs = [env if not d else None for env, d in zip(envs, dones)]
        envs = list(filter(lambda env: env is not None, envs))
        states = [state if not d else None for state, d in zip(states, dones)]
        states = np.array(list(filter(lambda state: state is not None, states)))
    return sum(returns) / tests


def train(net, opt):
    """
    :param net: tf.keras.Model
    :param opt: tf.keras.optimizers.Optimizer
    :return: float: mean of losses
    """
    t = tqdm(enumerate(ds_batch_generator()),  total=48 * 1e6 // batch_size, desc='loss: 10.0\t')
    loss = 10
    for i, (states, target_rewards) in t:
        t.set_description(f'loss: {round(float(loss), 5)}\t')

        with tf.GradientTape() as tape:
            reward_pred = net(states, training=True)
            loss = tf.reduce_mean(tf.square(target_rewards - reward_pred))

        grads = tape.gradient(target=loss, sources=net.trainable_variables)
        opt.apply_gradients(zip(grads, net.trainable_variables))


def get_file(idx, q=None):
    with open(f'datasets_3d/dataset_{idx}.pkl', 'rb') as file:
        data = pickle.load(file)

    states = np.concatenate([data[0][j].reshape((1, 16, 30, 2)) for j in range(len(data[0]))], axis=0)
    reward_states = np.concatenate([data[1][j].reshape((1, 480)) for j in range(len(data[0]))], axis=0)

    if q:
        q.put([states, reward_states])
    else:
        return states, reward_states


def ds_batch_generator():
    """
    Generates batches from dataset files
    """
    idxs = np.arange(48)
    np.random.shuffle(idxs)
    for idx in idxs:
        states, reward_states = get_file(idx)
        indices = np.arange(len(states))
        batch = []
        np.random.shuffle(indices)
        for j in indices:
            batch.append(j)
            if len(batch) == batch_size:
                yield states[batch], reward_states[batch]
                batch = []


def batch_generator(num_batches, parallel_size=10000):
    """
    Generate new training batches
    :param num_batches: int: number of batches to generate
    :param parallel_size: int:
    """
    total_batches = 0
    states = []
    rewards = []
    while True:
        envs = [TrainEnv(size, 99) for _ in range(parallel_size)]
        [env.reset() for env in envs]
        current_size = parallel_size
        dones = [False]
        while not all(dones):
            actions_b = np.random.randint(0, num_actions, (current_size,))
            envs, next_states, reward_b, dones = mp_act(envs, actions_b)
            for s, r in zip(next_states, reward_b):
                states.append(s)
                rewards.append(r)
                if len(states) == batch_size:
                    yield np.array(states), np.array(rewards)
                    states, rewards = [], []
                    total_batches += 1
                    if num_batches == total_batches:
                        return
            envs = [env if not d else None for env, d in zip(envs, dones)]
            envs = list(filter(lambda env: env is not None, envs))
            current_size = len(envs)


def create_datasets(num_files, file_size=1_000_000, parallel_size=20_000):
    """
    Create dataset files for training. To collect good data, i only save the first state and n random steps after that
    as the reward function only changes small amounts
    :param num_files: int: number of files to generate
    :param file_size: int: number of training pairs to collect per dataset
    :param parallel_size: int:
    """
    def store_and_save(new_states, new_rewards, states_list, rewards_list, file_idx):
        for s, r in zip(new_states, new_rewards):
            states_list.append(s)
            rewards_list.append(r)
            if len(states_list) == file_size:
                states_array = np.array(states_list, dtype=np.float32)
                rewards_array = np.array(rewards_list, dtype=bool)
                with open(f'datasets_3d/dataset_{file_idx}.pkl', 'wb') as file:
                    pickle.dump([states_array, rewards_array], file)
                states_list, rewards_list = [], []
                file_idx += 1
                print(file_idx)
        return states_list, rewards_list, file_idx
    states = []
    rewards = []
    num_loops = num_files * (file_size // (2 * parallel_size))
    file_num = 23
    envs = [TrainEnv(size, 99) for _ in range(parallel_size)]
    print(num_loops)
    for _ in range(num_loops):
        envs = mp_reset(envs)
        actions_b = np.random.randint(0, num_actions, (parallel_size,))
        envs, states_b, reward_b, _ = mp_act(envs, actions_b)
        states, rewards, file_num = store_and_save(states_b, reward_b, states, rewards, file_num)
        n_batch = [np.random.randint(1, len(np.where(reward_state == 1)[0])) for reward_state in reward_b]
        envs, states_b, reward_b = mp_n_moves(envs, n_batch)
        states, rewards, file_num = store_and_save(states_b, reward_b, states, rewards, file_num)


def reset_envs(envs):
    """
    used for multiprocessing
    :param envs: list: environments to reset
    :return: list: reset environments
    """
    [env.reset() for env in envs]
    return envs


def mp_reset(envs, processes=10):
    """
    :param envs: list: environments to reset
    :param processes: int: multiprocessing processes
    :return: list: reset environments
    """
    if len(envs) > 100:
        envs = np.array_split(np.array(envs), processes)
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(reset_envs, [env for env in envs])
        envs = np.concatenate(results).tolist()
    else:
        envs = reset_envs(envs)
    return envs


def make_n_moves(args):
    """
    :param args: list[list]: list of environments and list of int n moves to perform
    :return:
    """
    envs, n_batch = args
    [env.make_n_correct_moves(n) for env, n in zip(envs, n_batch)]
    states = np.vstack([env.get_state() for env in envs])
    rewards = np.vstack([env.get_state_rewards() for env in envs])
    return envs, states, rewards


def mp_n_moves(envs, n_batch, processes=10):
    """
    :param envs: list: environments
    :param n_batch: list[int]: number of steps to perform
    :param processes:
    :return:
    """
    if len(envs) > 100:
        envs = np.array_split(np.array(envs), processes)
        n_batch = np.array_split(n_batch, processes)
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(make_n_moves, [(env, n) for env, n in zip(envs, n_batch)])
        envs = np.concatenate([results[j][0] for j in range(processes)]).tolist()
        states = np.concatenate([results[j][1] for j in range(processes)], axis=0)
        state_rewards = np.concatenate([results[j][2] for j in range(processes)], axis=0)
    else:
        envs, states, state_rewards = make_n_moves((envs, n_batch))
    return envs, states, state_rewards


def take_actions(args):
    """
    multiprocessing environment actions
    :param args: list[list]: list of environments, list of actions to perform
    :return:
    """
    envs, actions = args
    [env.step(a) for env, a in zip(envs, actions)]
    next_states = np.vstack([env.get_state() for env in envs])
    rewards = np.vstack([env.get_state_rewards() for env in envs])
    dones = np.vstack([env.done for env in envs])
    return envs, next_states, rewards, dones


def mp_act(envs, actions, processes=12):
    """
    :param envs: list: environments
    :param actions: list[int]: list of actions to perform
    :param processes:
    :return:
    """
    if len(envs) > 100:
        envs = np.array_split(np.array(envs), processes)
        actions = np.array_split(actions, processes)
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(take_actions, [[env, action] for env, action in zip(envs, actions)])
        envs = np.concatenate([results[j][0] for j in range(processes)]).tolist()
        next_states = np.concatenate([results[j][1] for j in range(processes)], axis=0)
        rewards = np.concatenate([results[j][2] for j in range(processes)], axis=0)
        dones = np.concatenate([results[j][3] for j in range(processes)], axis=0)
    else:
        envs, next_states, rewards, dones = take_actions((envs, actions))
    return envs, next_states, rewards, dones


if __name__ == '__main__':
    # create_datasets(200)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

    inp = tf.keras.layers.Input(shape=(16, 30, 2))
    x = tf.keras.layers.Conv2D(128, 5, padding='same', activation=tf.keras.activations.swish)(inp)
    channels = [128, 256, 512, 1024, 2048]
    for i in range(len(channels) - 1):
        res = x
        x = tf.keras.layers.Conv2D(channels[i], 3, padding='same', activation=tf.keras.activations.swish)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(channels[i], 3, padding='same', activation=tf.keras.activations.swish)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([res, x])
        x = tf.keras.layers.Conv2D(channels[i + 1], 3, 2, padding='same', activation=tf.keras.activations.swish)(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation=tf.keras.activations.swish)(x)
    x = tf.keras.layers.Dense(1024, activation=tf.keras.activations.swish)(x)
    x = tf.keras.layers.Dense(num_actions, activation='sigmoid')(x)

    network = tf.keras.models.Model(inp, x)

    print(network.summary())

    network = tf.keras.models.load_model('Saves/DRN_Xbig_s.h5')

    score = test(network, tests=1000)  # board size [30, 16] and 99 mines, expert level

    print(score)
    for _ in range(1000):
        train(network, optimizer)
        new_score = test(network, tests=1000)
        print(new_score)
        if new_score > score:
            network.save('Saves/DRN_xbig_s.h5')
            old_score = score
