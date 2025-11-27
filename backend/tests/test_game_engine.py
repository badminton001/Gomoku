from backend.models.game_engine import GameEngine

# 测试：同一玩家（黑子）在一行连下五子，应该判定黑子获胜
def test_black_five_in_row():
    game = GameEngine(size=15, first_player=1)

    for x in range(5):
        ok = game.make_move_for(x, 0, player=1)
        assert ok, f"在 ({x}, 0) 落子失败"

    assert game.game_over is True, " game_over == True"
    assert game.winner == 1, "winner = 1"

    status = game.get_status()
    print("[test_black_five_in_row] PASS")
    print("  winner    =", status["winner"])
    print("  game_over =", status["game_over"])
    print("  history   =", status["move_history"])

# 测试：游戏结束后再次落子应该失败
def test_illegal_move_after_game_over():
    game = GameEngine(size=15, first_player=1)

    # 先造一个黑子五连
    for x in range(5):
        ok = game.make_move_for(x, 0, player=1)
        assert ok

    assert game.game_over is True
    assert game.winner == 1

    # 再尝试落子，应该返回 False，并且不改变结果
    ok = game.make_move(7, 7)
    assert ok is False
    assert game.winner == 1
    assert game.game_over is True

    print("[test_illegal_move_after_game_over] PASS")

# 测试：使用 make_move 时，current_player 是否在 1 和 2 之间正确切换
def test_turn_switching():
    game = GameEngine(size=15, first_player=1)

    # 初始应该是 1
    assert game.current_player == 1

    ok = game.make_move(7, 7)
    assert ok is True
    assert game.current_player == 2

    ok = game.make_move(8, 7)
    assert ok is True
    assert game.current_player == 1

    # 棋盘不应该结束
    assert game.game_over is False
    assert game.winner == 0

    print("[test_turn_switching] PASS")

# 测试：reset_game 是否能正确重置一局，但保留统计信息
def test_reset_game():
    game = GameEngine(size=15, first_player=1)

    # 先让黑子赢一局
    for x in range(5):
        game.make_move_for(x, 0, player=1)
    assert game.game_over is True
    assert game.winner == 1

    # 统计信息应该更新
    assert game.total_games == 1
    assert game.black_wins == 1
    assert game.white_wins == 0
    assert game.draws == 0

    # 重置游戏
    game.reset_game()

    # 重置后应该是一个全新的对局
    assert game.game_over is False
    assert game.winner == 0
    assert game.board.move_count == 0
    assert game.move_history == []
    assert game.current_player == game.first_player == 1

    # 但统计数据应该保留
    assert game.total_games == 1
    assert game.black_wins == 1

    print("[test_reset_game] PASS")

# 顺序跑所有测试
if __name__ == "__main__":
    test_black_five_in_row()
    test_illegal_move_after_game_over()
    test_turn_switching()
    test_reset_game()
    print("PASS!")

