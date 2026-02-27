# Standard Model Performance Testing Framework

本目录包含两个标准模型的性能测试脚本，用于评估TensorFlow MUSA PluggableDevice的整网性能。

## 目录结构

```
standard_model/
├── prunedGraph/          # 剪枝后的图模型 (GraphDef格式)
│   ├── run_graph_tf_musa.py    # 主要测试脚本
│   └── logs/                   # 日志和trace文件输出目录
└── wukong/               # Wukong深度学习模型
    ├── run_wukong_tf_musa.py   # 主要测试脚本  
    └── logs/                   # 日志和trace文件输出目录
```

## 通用命令行参数

两个模型脚本支持相同的命令行参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `musa` | 运行设备：`cpu` 或 `musa` |
| `--batch-size` | `100`(prunedGraph) / `1024`(wukong) | 批次大小 |
| `--warmup-rounds` | `5` | 预热轮数 |
| `--inference-rounds` | `20` | 推理轮数 |
| `--musa-plugin` | `../../tensorflow_musa_extension/build/libmusa_plugin.so` | MUSA插件路径 |

## 运行模式

### 1. 整网推理性能测试 (推荐)

```bash
# prunedGraph模型
python prunedGraph/run_graph_tf_musa.py --inference-only

# wukong模型  
python wukong/run_wukong_tf_musa.py --inference-only
```

**输出位置**: 
- `prunedGraph/logs/graph_inference/YYYY-MM-DD-HH.MM.SS_trace/`
- `wukong/logs/tensorflow_inference/YYYY-MM-DD-HH.MM.SS_trace/`

**生成文件**:
- `inference_only_result_MUSA_*.json` - 性能结果
- `inference_only_result_CPU_*.json` - CPU对比结果

### 2. 算子级性能分析

```bash
# prunedGraph模型
python prunedGraph/run_graph_tf_musa.py --profile-ops

# wukong模型
python wukong/run_wukong_tf_musa.py --profile-ops
```

**输出位置**: 同上，但会额外生成:
- `ops_profile_MUSA_*/` - TensorFlow Profiler trace文件
- `operator_timings_*.json` - 算子执行时间统计

### 3. 完整分析模式

```bash
# 不加任何特殊参数，默认运行完整分析
python prunedGraph/run_graph_tf_musa.py
python wukong/run_wukong_tf_musa.py
```

**包含内容**:
- 整网性能分析
- 算子级性能分析  
- 模型结构分析 (wukong)
- 设备信息分析

## 性能对比建议

1. **CPU vs MUSA对比**: 分别在`--device cpu`和`--device musa`下运行相同配置
2. **不同Batch Size**: 测试不同batch size对性能的影响
3. **多次运行取平均**: 由于系统负载波动，建议多次运行取平均值

## 输出文件说明

### 性能结果文件 (`inference_only_result_*.json`)
```json
{
  "device_type": "MUSA",
  "warmup_rounds": 5,
  "profiling_rounds": 20,
  "average_time": 0.04221096634864807,
  "min_time": 0.032160401344299316,
  "max_time": 0.06770586967468262,
  "average_throughput": 2425.909872667003,
  "batch_size": 1024
}
```

### 算子时间文件 (`operator_timings_*.json`)
```json
{
  "device_type": "MUSA",
  "operators": [
    {
      "name": "MatMul",
      "total_time_ms": 12.345,
      "count": 2,
      "avg_time_ms": 6.172
    }
  ]
}
```

## 故障排除

1. **MUSA插件加载失败**: 确保`tensorflow_musa_extension`已正确编译
2. **找不到模型文件**: 确保模型文件放在正确位置或使用`--model`参数指定
3. **内存不足**: 减小`--batch-size`参数值
4. **性能异常**: 检查是否有算子fallback到CPU执行