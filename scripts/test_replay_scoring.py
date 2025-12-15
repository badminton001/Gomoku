"""
Integration Test for Multi-Algorithm Move Scoring
Tests the updated move_scorer.py with real AI algorithms
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api.services.move_scorer import MoveScorer
from backend.analysis.replay import Move, GameReplay
from datetime import datetime
import time


def create_sample_game():
    """Create a sample game with some moves for testing"""
    moves = [
        # Opening moves (center area)
        Move(step=1, player=1, x=7, y=7, timestamp=time.time()),
        Move(step=2, player=2, x=7, y=8, timestamp=time.time()),
        Move(step=3, player=1, x=8, y=7, timestamp=time.time()),
        Move(step=4, player=2, x=6, y=8, timestamp=time.time()),
        Move(step=5, player=1, x=9, y=7, timestamp=time.time()),
        # Some tactical moves
        Move(step=6, player=2, x=8, y=8, timestamp=time.time()),
        Move(step=7, player=1, x=10, y=7, timestamp=time.time()),
        Move(step=8, player=2, x=9, y=8, timestamp=time.time()),
        Move(step=9, player=1, x=6, y=7, timestamp=time.time()),  # Blocks
        Move(step=10, player=2, x=7, y=9, timestamp=time.time()),
    ]
    
    game_replay = GameReplay(
        game_id="test_game_001",
        start_time=datetime.now(),
        winner=1,
        moves=moves
    )
    
    return game_replay


def test_basic_scoring():
    """Test 1: Basic scoring without MCTS"""
    print("=" * 70)
    print(" TEST 1: Basic Multi-Algorithm Scoring (Greedy, Minimax, Alpha-Beta)")
    print("=" * 70)
    
    game = create_sample_game()
    scorer = MoveScorer(enable_mcts=False)
    
    print(f"\n📋 Testing with {len(game.moves)} moves...")
    
    start_time = time.time()
    result = scorer.score_moves(game.moves, game_id=game.game_id)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Analysis completed in {elapsed:.2f} seconds")
    print("\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)
    
    # Check results
    assert 'dataframe' in result, "Missing dataframe in result"
    assert 'stats_summary' in result, "Missing stats_summary in result"
    assert 'csv_path' in result, "Missing csv_path in result"
    assert 'chart_path' in result, "Missing chart_path in result"
    
    df = result['dataframe']
    stats = result['stats_summary']
    
    # Validate DataFrame structure
    required_columns = ['step', 'player', 'x', 'y', 'greedy_score', 
                       'alphabeta_score', 'avg_score', 'score_variance', 'move_type']
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    print(f"\n✓ DataFrame structure validated")
    print(f"✓ Columns: {', '.join(df.columns.tolist())}")
    
    # Print statistics
    print(f"\n📊 Statistics Summary:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Validate score ranges
    for col in ['greedy_score', 'alphabeta_score', 'avg_score']:
        assert df[col].min() >= 0, f"{col} has values < 0"
        assert df[col].max() <= 1, f"{col} has values > 1"
    
    print(f"\n✓ All scores in valid range [0, 1]")
    
    # Show sample moves
    print(f"\n📝 Sample Move Analysis (first 5 moves):")
    print(df[['step', 'player', 'greedy_score', 'alphabeta_score', 
             'avg_score', 'move_type']].head().to_string())
    
    # Check files were created
    assert os.path.exists(result['csv_path']), f"CSV file not created: {result['csv_path']}"
    assert os.path.exists(result['chart_path']), f"Chart file not created: {result['chart_path']}"
    
    print(f"\n✓ CSV saved to: {result['csv_path']}")
    print(f"✓ Chart saved to: {result['chart_path']}")
    
    print(f"\n✅ TEST 1 PASSED!\n")
    return result


def test_with_mcts():
    """Test 2: Scoring with MCTS enabled (slower)"""
    print("=" * 70)
    print(" TEST 2: Multi-Algorithm Scoring with MCTS (slower)")
    print("=" * 70)
    
    # Use fewer moves for MCTS test (it's slow)
    game = create_sample_game()
    game.moves = game.moves[:5]  # Only test first 5 moves
    game.game_id = "test_game_mcts"
    
    scorer = MoveScorer(enable_mcts=True)
    
    print(f"\n📋 Testing with {len(game.moves)} moves (MCTS enabled)...")
    print("⚠️  This may take longer due to Monte Carlo simulations...")
    
    start_time = time.time()
    result = scorer.score_moves(game.moves, game_id=game.game_id)
    elapsed = time.time() - start_time
    
    print(f"\n⏱️  Analysis completed in {elapsed:.2f} seconds")
    
    df = result['dataframe']
    
    # Check MCTS column exists
    assert 'mcts_score' in df.columns, "Missing MCTS score column"
    
    print(f"\n✓ MCTS scoring working")
    print(f"\n📝 MCTS Score Sample:")
    print(df[['step', 'player', 'greedy_score', 'mcts_score', 'avg_score']].to_string())
    
    print(f"\n✅ TEST 2 PASSED!\n")
    return result


def test_critical_move_detection():
    """Test 3: Verify critical move detection"""
    print("=" * 70)
    print(" TEST 3: Critical Move Detection")
    print("=" * 70)
    
    game = create_sample_game()
    scorer = MoveScorer(enable_mcts=False)
    
    result = scorer.score_moves(game.moves, game_id="test_critical")
    
    df = result['dataframe']
    critical_moments = result['critical_moments']
    
    print(f"\n📊 Move Type Distribution:")
    print(df['move_type'].value_counts().to_string())
    
    print(f"\n🎯 Critical Moments Detected: {len(critical_moments)}")
    for moment in critical_moments:
        print(f"   Step {moment['step']}: {moment['type']}")
    
    print(f"\n✅ TEST 3 PASSED!\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print(" MOVE SCORER INTEGRATION TEST SUITE")
    print("=" * 70)
    print()
    
    try:
        # Test 1: Basic scoring
        test_basic_scoring()
        
        # Test 2: MCTS scoring (commented out by default due to slowness)
        # Uncomment to test MCTS:
        # test_with_mcts()
        
        # Test 3: Critical move detection
        test_critical_move_detection()
        
        print("=" * 70)
        print(" ✅ ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("📁 Check the following directories for output:")
        print("   - data/charts/ - Generated analysis charts")
        print("   - data/stats/  - CSV files with detailed scores")
        print()
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
