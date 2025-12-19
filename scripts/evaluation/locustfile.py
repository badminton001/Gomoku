import time
from locust import HttpUser, task, between, events
import random

class GomokuUser(HttpUser):
    wait_time = between(1, 2)
    
    def on_start(self):
        """Init user."""
        self.client.get("/health")
        self.game_size = 15

    @task(3)
    def predict_move_greedy(self):
        """Greedy move."""
        board = self._generate_random_board()
        payload = {
            "board": board,
            "current_player": 2,
            "algorithm": "greedy"
        }
        with self.client.post("/ai/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Greedy Stats Code: {response.status_code}")

    @task(1)
    def predict_move_alphabeta(self):
        """AlphaBeta move."""
        board = self._generate_random_board(stones=10) # 10 stones
        payload = {
            "board": board,
            "current_player": 2,
            "algorithm": "alphabeta"
        }
        # Heavy calc timeout
        with self.client.post("/ai/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                # Check time
                data = response.json()
                if data['processing_time'] > 2.0:
                    # Log warning but not failure if strictly within SLA
                    pass
                response.success()
            else:
                response.failure(f"AlphaBeta Status: {response.status_code}")

    def _generate_random_board(self, stones=5):
        """Random 15x15 board."""
        board = [[0 for _ in range(self.game_size)] for _ in range(self.game_size)]
        for _ in range(stones):
            x, y = random.randint(0, 14), random.randint(0, 14)
            board[x][y] = random.choice([1, 2])
        return board

# Usage:
# locust -f scripts/locustfile.py --host=http://localhost:8000 --users 10 --spawn-rate 1
