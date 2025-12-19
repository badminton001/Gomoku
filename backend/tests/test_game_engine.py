from backend.models.game_engine import GameEngine

# Test: Black wins with 5 in a row
    game = GameEngine(size=15, first_player=1)

    for x in range(5):
        ok = game.make_move_for(x, 0, player=1)
        assert ok, f"Failed to play at ({x}, 0)"

    assert game.game_over is True, "game_over should be True"
    assert game.winner == 1, "winner should be 1"

    status = game.get_status()
    print("[test_black_five_in_row] PASS")
    print("  winner    =", status["winner"])
    print("  game_over =", status["game_over"])
    print("  history   =", status["move_history"])

# Test: Move illegal after game over
    game = GameEngine(size=15, first_player=1)

    # Create a 5-in-a-row for Black
    for x in range(5):
        ok = game.make_move_for(x, 0, player=1)
        assert ok

    assert game.game_over is True
    assert game.winner == 1

    # Attempt to play again, should return False and not change state
    ok = game.make_move(7, 7)
    assert ok is False
    assert game.winner == 1
    assert game.game_over is True

    print("[test_illegal_move_after_game_over] PASS")

# Test: Turn switching
    game = GameEngine(size=15, first_player=1)

    # Initial should be 1
    assert game.current_player == 1

    ok = game.make_move(7, 7)
    assert ok is True
    assert game.current_player == 2

    ok = game.make_move(8, 7)
    assert ok is True
    assert game.current_player == 1

    # Game should not be over
    assert game.game_over is False
    assert game.winner == 0

    print("[test_turn_switching] PASS")

# Test: Game reset
    game = GameEngine(size=15, first_player=1)

    # Let Black win one game
    for x in range(5):
        game.make_move_for(x, 0, player=1)
    assert game.game_over is True
    assert game.winner == 1

    # Stats should update
    assert game.total_games == 1
    assert game.black_wins == 1
    assert game.white_wins == 0
    assert game.draws == 0

    # Reset game
    game.reset_game()

    # Should be a fresh game after reset
    assert game.game_over is False
    assert game.winner == 0
    assert game.board.move_count == 0
    assert game.move_history == []
    assert game.current_player == game.first_player == 1

    # Stats should persist
    assert game.total_games == 1
    assert game.black_wins == 1

    print("[test_reset_game] PASS")

# Run all tests sequentially
if __name__ == "__main__":
    test_black_five_in_row()
    test_illegal_move_after_game_over()
    test_turn_switching()
    test_reset_game()
    print("PASS!")
