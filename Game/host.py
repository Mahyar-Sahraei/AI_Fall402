from player import Ai

n = 5
ai1 = Ai((n, n))
ai2 = Ai((n, n))
state = []
turn = 1
end = 2 * n * (n + 1)

while end != 0:
    if turn == 1:
        print("AI 1:")
        line = ai1.decide(state)
        state.append(line)
        ai1.print()
        if ai1.pass_move:
            turn = 1
        else:
            turn = 2
    else:
        print("AI 2:")
        line = ai2.decide(state)
        state.append(line)
        ai2.print()
        if ai2.pass_move:
            turn = 2
        else:
            turn = 1
    end -= 1
