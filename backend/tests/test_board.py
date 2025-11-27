from backend.models.board import Board

def test_horizontal_win():
    b = Board()
    for x in range(5):
        ok = b.place_stone(x, 0, player=1)
        assert ok
    result, line = b.get_game_result(with_line=True)
    assert result == 1
    assert line == [(x, 0) for x in range(5)]
    print("[test_horizontal_win] PASS")

def test_vertical_win():
    b = Board()
    for y in range(5):
        ok = b.place_stone(0, y, player=2)
        assert ok
    result, line = b.get_game_result(with_line=True)
    assert result == 2
    assert line == [(0, y) for y in range(5)]
    print("[test_vertical_win] PASS")


def test_diag_down_right_win():
    b = Board()
    for t in range(5):
        ok = b.place_stone(t, t, player=1)
        assert ok
    result, line = b.get_game_result(with_line=True)
    assert result == 1
    assert line == [(t, t) for t in range(5)]
    print("[test_diag_down_right_win] PASS")


def test_diag_down_left_win():
    b = Board()
    coords = [(4 - t, t) for t in range(5)]
    for (x, y) in coords:
        ok = b.place_stone(x, y, player=2)
        assert ok
    result, line = b.get_game_result(with_line=True)
    assert result == 2
    assert line == coords
    print("[test_diag_down_left_win] PASS")

# 同一位置落两次子，第二次应该失败
def test_illegal_move():
    b = Board()
    ok1 = b.place_stone(7, 7, player=1)
    ok2 = b.place_stone(7, 7, player=2)
    assert ok1 is True, "第一次落子应该成功"
    assert ok2 is False, "第二次在同一位置落子应该失败"
    print("[test_illegal_move] PASS")

# 初始棋盘游戏应为进行中
def test_game_not_over_initially():
    b = Board()
    result, line = b.get_game_result(with_line=True)
    assert result == 0
    assert line == [(-1, -1)]
    print("[test_game_not_over_initially] PASS")


if __name__ == "__main__":
    test_horizontal_win()
    test_vertical_win()
    test_diag_down_right_win()
    test_diag_down_left_win()
    test_illegal_move()
    test_game_not_over_initially()
    print("PASS!")
