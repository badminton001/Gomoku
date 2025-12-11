# 并行AI评估系统使用说明

## 概述

并行评估系统允许同时运行多个独立的评估批次，大幅提升评估速度。相比顺序执行，4批次并行可将评估时间从4小时缩短到1小时。

## 快速开始

### 1. 运行单个批次（测试）

```bash
python scripts/parallel_eval.py --batch-id 1 --total-batches 4 --games-per-batch 5
```

### 2. 并行运行多个批次

**Windows PowerShell**:
```powershell
# 启动4个独立窗口并行运行
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts/parallel_eval.py --batch-id 1 --total-batches 4 --games-per-batch 5"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts/parallel_eval.py --batch-id 2 --total-batches 4 --games-per-batch 5"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts/parallel_eval.py --batch-id 3 --total-batches 4 --games-per-batch 5"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts/parallel_eval.py --batch-id 4 --total-batches 4 --games-per-batch 5"
```

### 3. 合并结果

等待所有批次完成后：
```bash
python scripts/parallel_eval.py --merge
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch-id` | 批次编号（1开始） | 必填 |
| `--total-batches` | 总批次数 | 4 |
| `--games-per-batch` | 每批次每AI配对的对局数 | 5 |
| `--algorithms` | 要测试的AI列表 | Greedy Minimax-D2 AlphaBeta-D2 MCTS-300 DQN |
| `--merge` | 合并所有批次结果 | - |

## 常用配置

### 快速测试（20分钟）
```bash
# 2批次，每批次2局
python scripts/parallel_eval.py --batch-id 1 --total-batches 2 --games-per-batch 2
python scripts/parallel_eval.py --batch-id 2 --total-batches 2 --games-per-batch 2
python scripts/parallel_eval.py --merge
```

### 标准评估（1小时）
```bash
# 4批次，每批次5局（默认配置）
# 并行运行4个批次
python scripts/parallel_eval.py --merge  # 完成后执行
```

### 完整评估（2小时）
```bash
# 4批次，每批次10局
--total-batches 4 --games-per-batch 10
```

## 自定义AI组合

```bash
# 只测试MCTS和Greedy
python scripts/parallel_eval.py --batch-id 1 --algorithms MCTS-300 Greedy

# 测试所有AI（默认）
python scripts/parallel_eval.py --batch-id 1 --algorithms Greedy Minimax-D2 AlphaBeta-D2 MCTS-300 DQN
```

## 文件说明

### 批次文件
位置：`data/results/self_play/batch_X.json`

每个批次运行完成后生成，包含：
- 批次信息（ID、时间戳）
- 所有对局详细结果
- AI配置

### 合并结果
位置：`data/results/self_play/merged/`

包含两个文件：
- `merged_results_YYYYMMDD_HHMMSS.json` - 详细数据
- `merged_results_YYYYMMDD_HHMMSS.csv` - 表格数据

## 工作流程示例

### 场景1：单机顺序运行
```bash
# 依次运行4个批次
python scripts/parallel_eval.py --batch-id 1 --total-batches 4 --games-per-batch 5
python scripts/parallel_eval.py --batch-id 2 --total-batches 4 --games-per-batch 5
python scripts/parallel_eval.py --batch-id 3 --total-batches 4 --games-per-batch 5
python scripts/parallel_eval.py --batch-id 4 --total-batches 4 --games-per-batch 5

# 合并
python scripts/parallel_eval.py --merge
```

### 场景2：单机并行运行（推荐）
```powershell
# 同时启动4个进程
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts/parallel_eval.py --batch-id 1 --total-batches 4 --games-per-batch 5"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts/parallel_eval.py --batch-id 2 --total-batches 4 --games-per-batch 5"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts/parallel_eval.py --batch-id 3 --total-batches 4 --games-per-batch 5"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python scripts/parallel_eval.py --batch-id 4 --total-batches 4 --games-per-batch 5"

# 等待全部完成后合并
python scripts/parallel_eval.py --merge
```

### 场景3：多机分布式运行
**机器1-4分别运行**:
```bash
# 机器1
python scripts/parallel_eval.py --batch-id 1 --total-batches 4 --games-per-batch 5

# 机器2  
python scripts/parallel_eval.py --batch-id 2 --total-batches 4 --games-per-batch 5

# 机器3
python scripts/parallel_eval.py --batch-id 3 --total-batches 4 --games-per-batch 5

# 机器4
python scripts/parallel_eval.py --batch-id 4 --total-batches 4 --games-per-batch 5
```

**汇总机器**:
```bash
# 将所有batch_*.json文件复制到data/results/self_play/
# 然后执行合并
python scripts/parallel_eval.py --merge
```

## 常见问题

### Q: 批次运行失败怎么办？
A: 只需重新运行失败的批次，不影响其他已完成的批次：
```bash
python scripts/parallel_eval.py --batch-id 2 --total-batches 4 --games-per-batch 5
```

### Q: 如何查看进度？
A: 查看批次文件是否生成：
```bash
dir data\results\self_play\batch_*.json
```

### Q: 合并时找不到批次文件？
A: 确保批次文件在正确位置：
```
data/results/self_play/
├── batch_1.json
├── batch_2.json
├── batch_3.json
└── batch_4.json
```

### Q: 可以改变批次分配吗？
A: 可以，只要确保batch-id唯一且total-batches一致：
```bash
# 例如：8个批次，每批次更少游戏
--total-batches 8 --games-per-batch 2
```

### Q: DQN加载失败怎么办？
A: 系统会自动跳过DQN继续评估其他AI。如果需要DQN，需要先训练模型。

## 性能对比

| 配置 | 顺序执行 | 4批次并行 | 加速比 |
|------|---------|----------|-------|
| 200局评估 | ~4小时 | ~1小时 | 4x |
| 400局评估 | ~8小时 | ~2小时 | 4x |

## 注意事项

1. **批次ID从1开始**，不是0
2. **并行数量不要超过CPU核心数**，建议4-8个批次
3. **确保磁盘空间充足**，每批次约1-2MB
4. **重复运行会覆盖**，相同batch-id会覆盖之前的结果
5. **合并前检查批次完整性**，确保所有批次都已完成

## 输出示例

合并后的统计输出：
```
🏆 胜率统计:
   AlphaBeta-D2        : 100/120 =  83.3%
   Minimax-D2          : 100/120 =  83.3%
   Greedy              :  40/120 =  33.3%
   MCTS-300            :  20/120 =  16.7%
   DQN                 :  10/120 =   8.3%
```

## 下一步

评估完成后，可以使用分析工具：
```bash
# 生成详细分析报告
python scripts/analyze_performance.py

# 生成可视化图表
python scripts/generate_visualizations.py

# 生成完整报告
python scripts/generate_reports.py
```
