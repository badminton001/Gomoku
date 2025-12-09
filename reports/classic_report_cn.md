# Classic AI 优化对比（Alpha-Beta）

## 版本摘要
- **V1（初始实现）**：候选上限 12，默认启用评估缓存；节点排序含阻塞奖励；无深度自适应缩枝。
- **V2（本次优化）**：候选上限调至 10，深度>1 时再收紧 0.8 倍；可配置评估缓存（当前基准关闭，减少 `to_string()` 开销）；终局快速返回，避免无谓递归；保留统一启发式与阻塞奖励。

配置文件：`backend/config/ai_config.json`  
基准脚本：`python backend/benchmarks/run_classic_benchmark.py`（种子 42，10×10 中盘局面）。

## 基准结果对比
| 版本 | Depth | 候选数 | 探索节点 | 耗时 (ms) | 走法 | 启发式质量 |
|------|-------|--------|----------|-----------|------|------------|
| V1   | 3     | 44     | 228      | 156.848   | (4,4) | 5201.60 |
| V2   | 3     | 44     | 142      | 125.498   | (4,4) | 5201.60 |

## 结论
- **速度**：节点数减少约 37.7%，耗时下降约 20%，在相同深度下获得更快响应。
- **准确性**：最佳着法与启发式评分保持一致（(4,4)，5201.6），未牺牲决策质量。

## 如何复现
```bash
python -m unittest discover backend/tests      # 确认算法正确性
python backend/benchmarks/run_classic_benchmark.py  # 获取当前基准
```

## 可调参数（位于 `ai_config.json`）
- `alpha_beta.candidate_limit`: 控制分支宽度，减小以提速，增大以提高强度。
- `alpha_beta.use_eval_cache`: True/False 切换缓存；缓存适合大搜索树，小局面可关闭以降低开销。
- `alpha_beta.depth` / `alpha_beta.distance`: 调节搜索深度与邻域半径。
