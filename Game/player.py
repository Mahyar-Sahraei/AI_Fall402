from queue import Queue
import math

class MASK:
    up    = 0b0001
    right = 0b0010
    down  = 0b0100
    left  = 0b1000


class PLAYER:
    me = 0
    opp = 1


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.value = None
        self.lines = 0
        self.nlines = 0
    
    def add_line(self, where, player):
        self.lines |= where
        self.nlines += 1
        if self.nlines == 4:
            self.value = player

    def remove_line(self, where):
        self.lines &= (~where)
        self.nlines -= 1
        self.value = None

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

class Ai:
    def __init__(self, shape):
        self.shape = shape 
        self.X = shape[0] - 1
        self.Y = shape[1] - 1

        self.moves = 0
        self.cells = [[Cell(i, j) for j in range(self.Y)] for i in range(self.X)]
        self.pass_move = False

        self.marked = 0
        self.current_score = 0
        self.criticals = set()
        self.chaining_cells = set()
        self.chains = []

        self.action_stack = []
        self.MAX_DEPTH = 7
        self.MAX_ACTIONS = 4

    def mapping(self, line):
        ret = []
        vertical = line[0][1] == line[1][1]
        if vertical:
            x = min(line[0][0], line[1][0])
            if line[0][1] != 0:
                y = line[0][1] - 1
                ret.append((x, y, MASK.right))
            if line[0][1] != self.Y:
                y = line[0][1]
                ret.append((x, y, MASK.left))
        else:
            y = min(line[0][1], line[1][1])
            if line[0][0] != 0:
                x = line[0][0] - 1
                ret.append((x, y, MASK.down))
            if line[0][0] != self.X:
                x = line[0][0]
                ret.append((x, y, MASK.up))
        return ret
    
    def mapping_inv(self, cell):
        lines = set()
        if not (cell.lines & MASK.up):
            lines.add(((cell.x, cell.y), (cell.x, cell.y + 1)))
        if not (cell.lines & MASK.right):
            lines.add(((cell.x, cell.y + 1), (cell.x + 1, cell.y + 1)))
        if not (cell.lines & MASK.down):
            lines.add(((cell.x + 1, cell.y + 1), (cell.x + 1, cell.y)))
        if not (cell.lines & MASK.left):
            lines.add(((cell.x + 1, cell.y), (cell.x, cell.y)))
        return lines

    def cell_adjacents(self, cell):
        adj = []
        if cell.x != 0 and not (cell.lines & MASK.up):
            adj.append(self.cells[cell.x - 1][cell.y])
        
        if cell.y != self.Y - 1 and not (cell.lines & MASK.right):
            adj.append(self.cells[cell.x][cell.y + 1])
        
        if cell.x != self.X - 1 and not (cell.lines & MASK.down):
            adj.append(self.cells[cell.x + 1][cell.y])
        
        if cell.y != 0 and not (cell.lines & MASK.left):
            adj.append(self.cells[cell.x][cell.y - 1])

        return adj

    def act(self, line, player):
        self.pass_move = False
        mapping = self.mapping(line)
        actions = []
        for x, y, mask in mapping:
            cell = self.cells[x][y]
            cell.add_line(mask, player)
            if cell.nlines == 4:
                self.criticals.discard(cell)
                self.pass_move = True
                self.marked += 1
                if cell.value == PLAYER.me:
                    self.current_score += 1
                else:
                    self.current_score -= 1
            elif cell.nlines == 3:
                self.criticals.add(cell)
                self.update_chaining_cells(cell)
            elif cell.nlines == 2:
                self.update_chaining_cells(cell)
            actions.append((x, y, mask))
        self.action_stack.append(actions)

    def undo_act(self):
        last_actions = self.action_stack.pop()
        for last_action in last_actions:
            x, y, mask = last_action
            cell = self.cells[x][y]
            if cell.nlines == 4:
                self.criticals.add(cell)
                self.marked -= 1
                if cell.value == PLAYER.me:
                    self.current_score -= 1
                else:
                    self.current_score += 1
            elif cell.nlines == 3:
                self.criticals.discard(cell)

            cell.remove_line(mask)

            if cell.nlines in (1, 2):
                self.update_chaining_cells(cell)

    def update_chaining_cells(self, changed_cell):
        if changed_cell.nlines == 2:
            self.chaining_cells.add(changed_cell)
        else:
            self.chaining_cells.discard(changed_cell)
 

    def update_chains(self):
        self.chains = []
        visited = set()
        for chaining_cell in self.chaining_cells:
            if chaining_cell in visited:
                continue
            visited.add(chaining_cell)
            queue = Queue()
            queue.put(chaining_cell)
            chain = {chaining_cell}
            while not queue.empty():
                cell = queue.get()
                adjacents = self.cell_adjacents(cell)
                for adj in adjacents:
                    if adj not in chain:
                        chain.add(adj)
                        queue.put(adj)
                        visited.add(adj)
            self.chains.append(chain)
        self.chains.sort(key=lambda x: len(x))
    
    def some_actions(self, max_actions):
        actions = set()
        for cell in self.criticals:
            actions = actions.union(self.mapping_inv(cell))
        for row in self.cells:
            for cell in row:
                if cell.nlines < 2:
                    actions = actions.union(self.mapping_inv(cell))
                    if len(actions) >= max_actions:
                        return actions
        self.update_chains()
        for chain in self.chains:
            for cell in chain:
                actions = actions.union(self.mapping_inv(cell))
                if len(actions) >= max_actions:
                    return actions
                
        return actions
    
    def best_move(self):
        best_move = None
        best_evaluation = -math.inf
        for line in self.some_actions(self.MAX_ACTIONS):
            self.act(line, PLAYER.me)
            evaluation = self.minimax(-math.inf, math.inf, PLAYER.opp, 0, self.MAX_DEPTH)
            #print(f"line: {line}\nevaluation: {evaluation}")
            self.undo_act()
            if evaluation > best_evaluation:
                best_evaluation = evaluation
                best_move = line
        return best_move

    
    def minimax(self, alpha, beta, player, depth, max_depth):
        if depth == max_depth or self.marked == self.X * self.Y:
            return self.heuristic()
        
        if player == PLAYER.me:
            if self.pass_move:
                self.pass_move = False
                return self.minimax(alpha, beta, PLAYER.opp, depth, max_depth)
            value = -math.inf
            for line in self.some_actions(self.MAX_ACTIONS):
                self.act(line, PLAYER.me)
                value = max(value, self.minimax(alpha, beta, PLAYER.opp, depth + 1, max_depth))
                self.undo_act()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            if self.pass_move:
                self.pass_move = False
                return self.minimax(alpha, beta, PLAYER.me, depth, max_depth)
            value = math.inf
            for line in self.some_actions(self.MAX_ACTIONS):
                self.act(line, PLAYER.opp)
                value = min(value, self.minimax(alpha, beta, PLAYER.me, depth + 1, max_depth))
                self.undo_act()
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    def heuristic(self):
        return self.current_score

    def decide(self, state):
        for i in range(self.moves, len(state)):
            self.act(state[i], PLAYER.opp)
        self.moves = len(state) + 1

        self.pass_move = False
        line = self.best_move()
        self.act(line, PLAYER.me)
        return line
    
    def print(self):
        for row in self.cells:
            for cell in row:
                print("+", end="")
                if cell.lines & MASK.up:
                    print("-", end="")
                else:
                    print(" ", end="")
            print("+")
            for cell in row:
                if cell.lines & MASK.left:
                    print("|", end="")
                else:
                    print(" ", end="")
                if cell.value is not None:
                    print(cell.value, end="")
                else:
                    print(" ", end="")
            if row[-1].lines & MASK.right:
                print("|")
            else:
                print(" ")
        for cell in self.cells[-1]:
            print("+", end="")
            if cell.lines & MASK.down:
                print("-", end="")
            else:
                print(" ", end="")
        print("+")
