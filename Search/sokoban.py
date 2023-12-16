#!/usr/bin/env python

##################################################
#                                                #
#  Sokoban Game Solution Finder,                 #
#      Using DFS, BFS and A* Search Algorithms   #
#                                                #
##################################################



__author__ = "Mahyar Sahraei"
__email__  = "sahraei19@gmail.com"

from queue import Queue
import heapq
from copy import deepcopy

import numpy as np


# Action space (Four directions)
Actions = {
    'r': np.array([0,  1]),
    'l': np.array([0, -1]),
    'u': np.array([-1, 0]),
    'd': np.array([1,  0])
}


# State representation
class State:
    def __init__(self, board2D, agent, prev_action, parent, stones=None):
        self.board2D = board2D
        self.agent = agent
        self.prev_action = prev_action
        self.parent = parent
        self.stones = stones if stones is not None else parent.stones

        self.f_cost = None
        self.g_cost = None
        self.h_cost = None

        self.id = hash(self.board2D.data.tobytes())

    def set_cost(self, g_cost, h_cost):
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost

    def __eq__(self, other):
        return  self.id == other.id

    def __lt__(self, other):
        return self.f_cost < other.f_cost


# A holder class for a star's heuristic
class qll:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.up = None
        self.down = None
        self.closed = False
    
    def linkr(self, other):
        self.right = other
        other.left = self
    
    def linkd(self, other):
        self.down = other
        other.up = self

    def close(self):
        self.closed = True

    def is_closed(self):
        return self.closed


# A class for all three search algorithms
class Solver:
    def __init__(self, board2D: np.ndarray):
        self.board2D = np.array(board2D)
        self.height, self.width = self.board2D.shape

        self.agent = np.array([x[0] for x in np.where(self.board2D == 'a')])

        boxes_x, boxes_y = np.where(self.board2D == 'b')
        self.boxes = [(boxes_x[i], boxes_y[i]) for i in range(len(boxes_x))]

        stones_x, stones_y = np.where(self.board2D == 's')
        self.stones = [(stones_x[i], stones_y[i]) for i in range(len(stones_x))]

        self.n_targets = len(self.stones)

        self.visit_map = {}
        self.heuristic_tree = None

        self.DECAY_RATE = -5.00
        self.GROWTH_RATE = 1.1
        self.H_FACTOR = 100
        self.STEP_COST = 0.1

    # DFS's function caller
    def dfs(self):
        return self._dfs(State(self.board2D, self.agent, None, None, self.stones))

    # DFS's implementation
    def _dfs(self, state: State):
        if self.is_goal(state):
            return self.trace_actions(state)

        self.visit(state)

        av_actions = self.get_actions(state)
        for act in av_actions:
            new_state = self.next_state(state, act)
            if not self.is_visited(new_state):
                result = self._dfs(new_state)
                if result is not None:
                    return result
        return None

    # BFS's implementation
    def bfs(self):
        state_queue = Queue()
        start_state = State(self.board2D, self.agent, None, None, self.stones)
        state_queue.put(start_state)
        while state_queue.not_empty:
            state = state_queue.get()

            if self.is_goal(state):
                return self.trace_actions(state)

            self.visit(state)
            av_actions = self.get_actions(state)
            for act in av_actions:
                new_state = self.next_state(state, act)
                if not self.is_visited(new_state):
                    state_queue.put(new_state)

        return None

    # A star's implementation
    def a_star(self):
        self.setup_heuristic_tree()
        start_state = State(self.board2D, self.agent, None, None, self.stones)
        open_set = [start_state]
        start_state.set_cost(0, self.heuristic(start_state))

        while len(open_set) != 0:
            current_state = heapq.heappop(open_set)
            if self.is_goal(current_state):
                return self.trace_actions(current_state)
            self.visit(current_state)

            av_actions = self.get_actions(current_state)
            for action in av_actions:
                new_state = self.next_state(current_state, action, True)

                if self.is_visited(new_state):
                    continue

                if new_state in open_set:
                    continue

                g_cost = new_state.parent.g_cost + self.STEP_COST
                h_cost = self.heuristic(new_state)
                new_state.set_cost(g_cost, h_cost)
                heapq.heappush(open_set, new_state)

        return None


    # Auxiliary functions
    def get_actions(self, state: State):
        actions = []
        y, x = state.agent

        if x > 0:
            if state.board2D[y, x - 1] in ('f', 'b'):
                actions.append('l')
            elif x > 1 and state.board2D[y, x - 1] != 'w':
                if state.board2D[y, x - 2] in ('f', 'b'):
                    actions.append('l')
        if x < self.width - 1:
            if state.board2D[y, x + 1] in ('f', 'b'):
                actions.append('r')
            elif x < self.width - 2 and state.board2D[y, x + 1] != 'w':
                if state.board2D[y, x + 2] in ('f', 'b'):
                    actions.append('r')
        if y > 0:
            if state.board2D[y - 1, x] in ('f', 'b'):
                actions.append('u')
            elif y > 1 and state.board2D[y - 1, x] != 'w':
                if state.board2D[y - 2, x] in ('f', 'b'):
                    actions.append('u')
        if y < self.height - 1:
            if state.board2D[y + 1, x] in ('f', 'b'):
                actions.append('d')
            elif y < self.height - 2 and state.board2D[y + 1, x] != 'w':
                if state.board2D[y + 2, x] in ('f', 'b'):
                    actions.append('d')

        return np.array(actions)

    def next_state(self, state: State, action: str, update_stones=False):
        new_agent = state.agent + Actions[action]
        new_board2D = state.board2D.copy()
        new_board2D[tuple(new_agent)] = 'a'
        new_board2D[tuple(state.agent)] = 'b' if tuple(state.agent) in self.boxes else 'f'
        new_stones = None

        if state.board2D[tuple(new_agent)] in ('s', 'x'):
            new_stone = tuple(new_agent + Actions[action])
            if state.board2D[new_stone] == 'b':
                new_board2D[new_stone] = 'x'
            else:
                new_board2D[new_stone] = 's'

            if update_stones:
                stones_x, stones_y = np.where(np.logical_or(new_board2D == 's', new_board2D == 'x'))
                new_stones = [(stones_x[i], stones_y[i]) for i in range(len(stones_x))]

        return State(new_board2D, new_agent, action, state, new_stones)


    def visit(self, state: State):
        self.visit_map[state.id] = True

    def is_visited(self, state: State):
        if self.visit_map.get(state.id):
            return True
        return False

    def is_goal(self, state: State):
        for box in self.boxes:
            if state.board2D[box] != 'x':
                return False
        return True

    def trace_actions(self, state: State):
        actions = []
        while state is not None:
            actions.append(state.prev_action)
            state = state.parent
        return actions[-2::-1]

    def heuristic(self, state: State):
        h_cost = 0
        for stone in state.stones:
            x, y = stone
            h_cost += self.heuristic_tree[x][y].val

        distance = lambda s: abs(s[0] - state.agent[0]) + abs(s[1] - state.agent[1])
        vector = np.array([distance(stone) for stone in state.stones if stone not in self.boxes])
        delta = 0
        if len(vector) != 0:
            weights = np.array([ np.exp(self.DECAY_RATE * v) for v in vector])
            delta = np.average(vector, weights=weights)

        return h_cost * self.H_FACTOR + delta

    def setup_heuristic_tree(self):
        h, w = self.board2D.shape
        self.heuristic_tree = [[qll(h * w * 10) for i in range(w)] for j in range(h)]
        for i in range(h):
            for j in range(w - 1):
                if self.board2D[i][j] != 'w' and self.board2D[i][j + 1] != 'w':
                    self.heuristic_tree[i][j].linkr(self.heuristic_tree[i][j + 1])

        for i in range(h - 1):
            for j in range(w):
                if self.board2D[i][j] != 'w' and self.board2D[i + 1][j] != 'w':
                    self.heuristic_tree[i][j].linkd(self.heuristic_tree[i + 1][j])

        heuristic_trees = []
        for box in self.boxes:
            heuristic_trees.append(deepcopy(self.heuristic_tree))
            queue = Queue()
            x, y = box
            heuristic_trees[-1][x][y].val = 1
            heuristic_trees[-1][x][y].close()
            queue.put(heuristic_trees[-1][x][y])
            while not queue.empty():
                current_qll = queue.get()

                if current_qll.right is not None:
                    if current_qll.right.right is not None and\
                        not current_qll.right.is_closed():
                        current_qll.right.val = current_qll.val * self.GROWTH_RATE
                        current_qll.right.close()
                        queue.put(current_qll.right)

                if current_qll.left is not None:
                    if current_qll.left.left is not None and\
                        not current_qll.left.is_closed():
                        current_qll.left.val = current_qll.val * self.GROWTH_RATE
                        current_qll.left.close()
                        queue.put(current_qll.left)

                if current_qll.up is not None:
                    if current_qll.up.up is not None and\
                        not current_qll.up.is_closed():
                        current_qll.up.val = current_qll.val * self.GROWTH_RATE
                        current_qll.up.close()
                        queue.put(current_qll.up)

                if current_qll.down is not None:
                    if current_qll.down.down is not None and\
                        not current_qll.down.is_closed():
                        current_qll.down.val = current_qll.val * self.GROWTH_RATE
                        current_qll.down.close()
                        queue.put(current_qll.down)

        for i in range(h):
            for j in range(w):
                vector = np.array([heuristic_trees[k][i][j].val - 1 for k in range(self.n_targets)])
                weights = np.array([np.exp(self.DECAY_RATE * v) for v in vector], dtype=float)
                if weights.sum() == 0:
                    self.heuristic_tree[i][j].val = vector.min()
                else:
                    self.heuristic_tree[i][j].val = np.average(vector, weights=weights)


    def animate(self, actions):
        from subprocess import call
        from time import sleep

        state = State(self.board2D, self.agent, None, None, self.stones)
        call("clear")
        print(state.board2D)
        sleep(1)
        for action in actions:
            state = self.next_state(state, action)
            call("clear")
            print(state.board2D)
            sleep(0.5)



# Main functions
def dfs(board2D):
    solver = Solver(board2D)
    return solver.dfs()

def bfs(board2D):
    solver = Solver(board2D)
    return solver.bfs()

def a_star(board2D):
    solver = Solver(board2D)
    return solver.a_star()
