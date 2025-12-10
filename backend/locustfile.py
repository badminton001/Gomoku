from locust import HttpUser, task, between

class GomokuUser(HttpUser):
    # 模拟每个用户思考 1 到 3 秒后再下子
    wait_time = between(1, 3)

    @task
    def predict_move(self):
        # 模拟发送一个标准棋盘给 AI
        payload = {
            "board": [[0]*15 for _ in range(15)], # 空棋盘
            "current_player": 1,
            "algorithm": "minimax"
        }
        # 发送 POST 请求轰炸接口
        self.client.post("/ai/predict", json=payload)