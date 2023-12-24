from queue import Queue


class Var:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y)

    def __hash__(self):
        return hash((self.x, self.y))



class Sudoku:
    def __init__(self, board):
        self.board_size = len(board)
        self.board = board
        self.domains = {}

        for i in range(self.board_size):
            for j in range(self.board_size):
                var = Var(i, j, self.board[i][j])
                if self.board[i][j] == 0:
                    self.domains[var] = set(range(1, self.board_size + 1))
                else:
                    self.domains[var] = set([self.board[i][j]])
    
    def reduce_domain(self, domain, index):
        if index == len(domain) - 1:
            domain.pop()
        else:
            domain[index] = domain.pop()
    
    def get_neighbours(self, var):
        neighbours = []
        for i in range(self.board_size):
            if i != var.y:
                x, y = var.x, i
                neighbours.append(Var(x, y, self.board[x][y]))
            if i != var.x:
                x, y = i, var.y
                neighbours.append(Var(x, y, self.board[x][y]))

        return neighbours
    
    def assigns(self, var, value):
        neighbours = self.get_neighbours(var)
        for neighbour in neighbours:
            if neighbour.value == value:
                return False
        return True
    
    def AC_3(self):
        queue = Queue()
        queued = {}

        for var in self.domains:
            for neighbour in self.get_neighbours(var):
                arc = (var, neighbour)
                queue.put(arc)
                queued[arc] = True

        while not queue.empty():
            var1, var2 = queue.get()
            dom1, dom2 = self.domains[var1], self.domains[var2]
                
            if len(dom1) == 1 and dom1.issubset(dom2):
                for value in dom1:
                    break
                dom2.remove(value)
                for neighbour in self.get_neighbours(var2):
                    arc = (var2, neighbour)
                    if not queued[arc]:
                        queue.put(arc)
                        queued[arc] = True

            queued[(var1, var2)] = False

    def backtrack(self):
        unsolved = []
        for var in self.domains:
            if len(self.domains[var]) == 1:
                for value in self.domains[var]:
                    break
                self.board[var.x][var.y] = value
            else:
                unsolved.append(var)

        return self._backtrack(unsolved)
    
    def _backtrack(self, unsolved):
        if len(unsolved) == 0:
            return self.board
        
        var = unsolved.pop()
        domain = self.domains[var]
        for value in domain:
            neighbours = self.get_neighbours(var)
            domain_changed_vars = []
            for neighbour in neighbours:
                if value in self.domains[neighbour]:
                    self.domains[neighbour].remove(value)
                    domain_changed_vars.append(neighbour)

            self.board[var.x][var.y] = value
            result = self._backtrack(unsolved)
            if result is not None:
                return result
            
            for dc_var in domain_changed_vars:
                self.domains[dc_var].add(value)

        self.board[var.x][var.y] = 0
        unsolved.append(var)
        return None

    def solve(self):
        self.AC_3()
        return self.backtrack()


# Main function
def solver(board):
    sudoku = Sudoku(board)
    return sudoku.solve()
