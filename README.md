# Gomoku AI

基于 AI 算法的五子棋游戏系统

## 技术栈

- 后端：FastAPI + Python
- 前端：Tkinter
- 数据可视化：matplotlib + seaborn
- 算法：numpy + pandas

## 快速开始

### 后端
```bash
cd backend
pip install -r requirements.txt
python main.py
```

## 经典博弈算法（Person C 交付）
- 路径：`backend/algorithms/classic_ai.py`
- 四种算法：随机、贪心（启发式）、Minimax、Alpha-Beta（统一评估函数与邻域候选）。
- 配置：`backend/config/ai_config.json` 可调搜索深度、候选数、邻域半径等超参数。
- 基准：`python backend/benchmarks/run_classic_benchmark.py` 输出耗时、候选数、节点数及启发式质量。

## 测试
```bash
python -m unittest discover backend/tests
```
