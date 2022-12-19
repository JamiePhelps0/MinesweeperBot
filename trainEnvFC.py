import random
import numpy as np
np.set_printoptions(linewidth=400)


class TrainEnv:
    def __init__(self, board_size, num_mines, verbose=False):
        self._state = None
        self.num_mines = num_mines
        self.size_x = board_size[0]
        self.size_y = board_size[1]
        self.mine_list = []
        self.full_board = None
        self.viewing_board = None
        self.verbose = verbose
        self.ep_steps = 0
        self.first_move = True
        self.done = False

    def check_pos(self, i, num, board):
        pos = []
        if 0 <= i[1] - 1 <= self.size_y - 1 and 0 <= i[0] - 1 <= self.size_x - 1:
            if board[i[1] - 1][i[0] - 1] in num:
                pos.append([i[0] - 1, i[1] - 1])
        if 0 <= i[1] - 1 <= self.size_y - 1 and 0 <= i[0] <= self.size_x - 1:
            if board[i[1] - 1][i[0]] in num:
                pos.append([i[0], i[1] - 1])
        if 0 <= i[1] - 1 <= self.size_y - 1 and 0 <= i[0] + 1 <= self.size_x - 1:
            if board[i[1] - 1][i[0] + 1] in num:
                pos.append([i[0] + 1, i[1] - 1])
        if 0 <= i[1] <= self.size_y - 1 and 0 <= i[0] - 1 <= self.size_x - 1:
            if board[i[1]][i[0] - 1] in num:
                pos.append([i[0] - 1, i[1]])
        if 0 <= i[1] <= self.size_y - 1 and 0 <= i[0] + 1 <= self.size_x - 1:
            if board[i[1]][i[0] + 1] in num:
                pos.append([i[0] + 1, i[1]])
        if 0 <= i[1] + 1 <= self.size_y - 1 and 0 <= i[0] - 1 <= self.size_x - 1:
            if board[i[1] + 1][i[0] - 1] in num:
                pos.append([i[0] - 1, i[1] + 1])
        if 0 <= i[1] + 1 <= self.size_y - 1 and 0 <= i[0] <= self.size_x - 1:
            if board[i[1] + 1][i[0]] in num:
                pos.append([i[0], i[1] + 1])
        if 0 <= i[1] + 1 <= self.size_y - 1 and 0 <= i[0] + 1 <= self.size_x - 1:
            if board[i[1] + 1][i[0] + 1] in num:
                pos.append([i[0] + 1, i[1] + 1])
        return pos

    def get_zeros(self, pos):
        unknown = [pos]
        zeros = [pos]
        while not unknown == []:
            new_unknowns = []
            for i in unknown:
                pos = self.check_pos(i, [0], self.full_board)
                for j in pos:
                    if j not in zeros:
                        zeros.append([j[0], j[1]])
                        new_unknowns.append([j[0], j[1]])
            unknown = new_unknowns
        return zeros

    def in_board(self, pos):
        if pos[1] < 0 or pos[1] > self.size_y - 1 or pos[0] < 0 or pos[0] > self.size_x - 1:
            return False
        return True

    def get_action_mask(self):
        action_mask = [False for _ in range(self.size_x * self.size_y)]
        for row in range(self.size_y):
            for column in range(self.size_x):
                if self.viewing_board[row][column] != -1:
                    action_mask[row * self.size_x + column] = True
        return np.array(action_mask, dtype=np.bool)

    def encode(self):
        # self._state = 0.2 * np.array(self.viewing_board, dtype=np.float32).reshape((1, self.size_y, self.size_x))

        self._state = np.array(self.viewing_board, dtype=np.float32).reshape((self.size_x * self.size_y,))
        self._state = np.array([[0 if i == -1 else 1, 0.125 * (i + 1)] for i in self._state], dtype=np.float32).reshape((1, self.size_y, self.size_x, 2))

    def get_state(self):
        return self._state

    def create_board(self, pos):
        self.full_board = [[0 for _ in range(self.size_x)] for _ in range(self.size_y)]
        self.viewing_board = [[-1 for _ in range(self.size_x)] for _ in range(self.size_y)]
        self.mine_list = []
        start_list = [[pos[0] - 1, pos[1] - 1], [pos[0] - 1, pos[1]], [pos[0] - 1, pos[1] + 1],
                      [pos[0], pos[1] - 1], [pos[0], pos[1]], [pos[0], pos[1] + 1],
                      [pos[0] + 1, pos[1] - 1], [pos[0] + 1, pos[1]], [pos[0] + 1, pos[1] + 1]]
        for i in range(self.num_mines):
            x, y = None, None
            first = True
            while ([x, y] in self.mine_list or [x, y] in start_list) or first:
                first = False
                x = random.randint(0, self.size_x - 1)
                y = random.randint(0, self.size_y - 1)
            self.full_board[y][x] = 10
            self.mine_list.append([x, y])
        for mine in self.mine_list:
            adj_squares = self.check_pos(mine, [0, 1, 2, 3, 4, 5, 6, 7], self.full_board)
            for square in adj_squares:
                self.full_board[square[1]][square[0]] += 1
        self.select_pos(pos)
        if self.verbose:
            print(np.array(self.full_board))

    def select_pos(self, pos):
        if self.first_move:
            self.first_move = False
            self.create_board(pos)
            return False
        if pos in self.mine_list:
            return True
        if not self.full_board[pos[1]][pos[0]] == 0:
            self.viewing_board[pos[1]][pos[0]] = self.full_board[pos[1]][pos[0]]
            return False
        zeros = self.get_zeros(pos)
        for i in zeros:
            self.viewing_board[i[1]][i[0]] = self.full_board[i[1]][i[0]]
            out = self.check_pos(i, [1, 2, 3, 4, 5, 6, 7, 8], self.full_board)
            for j in out:
                self.viewing_board[j[1]][j[0]] = self.full_board[j[1]][j[0]]
        return False

    def reset(self):
        if self.verbose:
            print('\n\n\n')
        self.ep_steps = 0
        self.first_move = True
        self.viewing_board = [[-1 for _ in range(self.size_x)] for _ in range(self.size_y)]
        self.encode()
        return self._state

    def guess(self, pos):
        if all(elem == -1 for elem in [self.viewing_board[i[1]][i[0]] for i in
                                       self.check_pos(pos, [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8], self.viewing_board)]):
            return True
        return False

    def get_state_rewards(self):
        def is_end(v, m):
            if (v != -1) or (m == 10):
                return True
            return False
        board = np.array(self.viewing_board).flatten()
        main_board = np.array(self.full_board).flatten()
        rewards = np.array([0 if is_end(v, m) else 1 for v, m in zip(board, main_board)])
        return rewards

    def make_n_correct_moves(self, n, reward_state=None):
        reward_state = reward_state if reward_state is not None else self.get_state_rewards()
        idxs = np.where(reward_state == 1)[0]
        actions = np.random.choice(idxs, min(n, len(idxs)), replace=False)
        for a in actions:
            action = [a % self.size_x, a // self.size_x]
            self.select_pos(action)
        self.encode()
        return len(idxs)

    def print_action(self, outcome, pos, reward):
        print('*******************************')
        print(np.array(self.viewing_board))
        print(outcome, pos[0], pos[1], reward, self.viewing_board)
        print('*******************************')

    def step(self, action):
        self.ep_steps += 1
        action = [action % self.size_x, action // self.size_x]
        if not self.first_move:
            useless_act = self.viewing_board[action[1]][action[0]] != -1
        else:
            useless_act = False

        end = self.select_pos(action)
        self.encode()

        if self.ep_steps >= self.size_x * self.size_y + 1:
            if self.verbose:
                self.print_action('Max Steps', action, -1)
            self.done = True
            return self._state, 0, True
        if np.count_nonzero(self.viewing_board == -1) == len(self.mine_list):
            if self.verbose:
                self.print_action('Done', action, 1)
            self.done = True
            return self._state, 1, True
        if useless_act:
            if self.verbose:
                self.print_action('Useless', action, -1)
            self.done = True
            return self._state, 0, True
        if end:
            if self.verbose:
                self.print_action('Mine', action, -1)
            self.done = True
            return self._state, 0, True
        else:
            if self.verbose:
                self.print_action('Good Action', action, 1)
            self.done = False
            return self._state, 1, False


if __name__ == '__main__':
    returns = []
    env = TrainEnv([30, 16], 99)
    for _ in range(500):
        rtrn = 1
        env.reset()
        env.step(255)
        state_rewards = env.get_state_rewards()
        while any(state_rewards == 1):
            env.make_n_correct_moves(1, reward_state=state_rewards)
            rtrn += 1
            state_rewards = env.get_state_rewards()
        returns.append(rtrn)
    print(sum(returns) / 100, min(returns), max(returns))


