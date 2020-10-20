"""
A 3x3 Jeu-de-tacquin puzzle Solver
"""

import sys
import heapq

WINNING_POS = "123456780"


class Position():
    """
    Represents a board state in the puzzle
    """
    def __init__(self, pos):
        self.pos = pos

    def __hash__(self):
        return hash(self.pos)

    def __eq__(self, other):
        return self.pos == other.pos

    def __iter__(self):
        return iter(self.pos)

    def __str__(self):
        ret = ""
        for i in range(0, 9, 3):
            line = ' '.join(val for val in self.pos[i:i+3])
            ret += line + '\n'
        return ret

    def cost(self):
        """
        The cost of the position, given by how many tiles are out of place
        Used as the heuristic for A* search
        """
        return sum(correct_tile != my_tile
                   for correct_tile, my_tile in zip(WINNING_POS, self))

    def get_moves(self):
        empty_space = self.pos.find('0')
        possible_moves = []

        # Left
        if empty_space % 3 != 0:
            possible_moves.append(empty_space - 1)

        # Above
        if empty_space < 6:
            possible_moves.append(empty_space + 3)

        # Right
        if empty_space % 3 != 2:
            possible_moves.append(empty_space + 1)

        # Below
        if empty_space > 2:
            possible_moves.append(empty_space - 3)

        for movable_tile in possible_moves:
            new_position = swap(self.pos, movable_tile, empty_space)
            yield Position(new_position)


class Node():
    """
    Represents a searchable node in the graph to be searched to find a solution
    """
    def __init__(self, position, path=None):
        self.position = position
        self.path = path or []
        self.cost = self.position.cost() + len(self.path)

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        path_str = "\n".join(str(pos) for pos in self.path)
        divider = '----\n'
        position_str = str(self.position)
        return path_str + divider + position_str


winning_position = Position(WINNING_POS)


def solve(starting_pos):
    """
    Solves a 3x3 Jeu-de-Tacquin puzzle from the given starting position
    """
    heap = [Node(starting_pos)]
    visited = set()

    while heap:
        node = heapq.heappop(heap)
        position, path = node.position, node.path

        visited.add(position)

        if position == winning_position:
            return node

        for move in position.get_moves():
            if move not in visited:
                move_path = path + [move]
                heapq.heappush(heap, Node(move, move_path))


def swap(string, ind1, ind2):
    """
    Swaps the characters of a string that are at ind1 and ind2
    """
    lst = list(string)
    lst[ind1], lst[ind2] = lst[ind2], lst[ind1]
    return "".join(lst)


def main():
    """
    Accepts a Jeu-de-Tacquin board from stdin and prints its solution
    """
    if len(sys.argv) != 2:
        print('usage: python npuzzle.py [starting_pos]')
        return

    starting_pos = sys.argv[1]
    if set(starting_pos) != set(WINNING_POS):
        print("""
        Enter a 3x3 Jeu-de-Tacquin board as a 9-character string, letting 0
        represent the empty space. For example:

        python npuzzle.py 573216840
        """)
        return

    solution = solve(Position(starting_pos))
    print(solution)


if __name__ == '__main__':
    main()
