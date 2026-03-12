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
    ├── test_tf_musa_extension.py   # 测试 tensorflow musa extension .so 文件是否能跑
    └── logs/                   # 日志和trace文件输出目录
```

## 测试模型说明

### 1. prunedGraph

`prunedGraph` 是一个已经固化好的 `GraphDef(.pb)` 模型，适合验证：

- 自定义 graph optimizer 是否被正确加载
- 融合 pattern 是否命中
- 融合后的自定义算子是否真的进入最终执行图
- CPU vs MUSA 的端到端精度是否一致

这个模型对图优化开发尤其有价值，因为它天然是图模式，便于：

- dump `before_fusion / after_fusion / final` 图
- 在 Netron 中直接观察子图是否被替换
- 检查融合算子是否在最终输出路径上可达

从当前模型和日志看，`prunedGraph` 中常见的算子类型包括：

- `MatMul` / `BatchMatMulV2`
- `BiasAdd`
- `AddV2` / `Mul` / `Sub` / `RealDiv`
- `Reshape` / `Transpose` / `Pack` / `ConcatV2`
- `GatherV2`
- `Mean` / `Prod` / `Select`
- `FusedBatchNormV3`
- `Sqrt` / `Rsqrt`
- 以及图优化命中后出现的融合算子，例如 `MusaGelu`

如果后续要开发 graph fusion，`prunedGraph` 是优先推荐的验证入口。

### 2. wukong

`wukong` 是一个 TensorFlow/Keras 模型脚本，适合验证：

- 动态模型在 MUSA 上是否能正常前向执行
- 整网推理性能
- eager / `tf.function` 路径下的运行情况

它更偏向“模型级功能与性能验证”，而不是严格的 GraphDef 融合可视化验证。

从模型结构和 profile 结果看，`wukong` 中常见的算子类型包括：

- `ResourceGather` / `ReadVariableOp`
- `MatMul` / `BatchMatMulV2`
- `FusedBatchNormV3`
- `Relu`
- `AddV2` / `Mul` / `Sub`
- `Reshape` / `Transpose` / `StridedSlice` / `Pack` / `Fill`
- `ConcatV2`
- `Rsqrt`
- `Send`（跨设备数据搬运事件，在 profiler 中可见）

需要注意的是，当前 `wukong` 模型本身使用的是 `ReLU`，不是 `GELU`，因此它不适合作为 `MusaGelu` 融合命中的验证模型；但它仍然适合验证插件加载、设备执行和整网性能。

## 通用命令行参数

两个模型脚本支持相同的命令行参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | `musa` | 运行设备：`cpu` 或 `musa` |
| `--batch-size` | `100`(prunedGraph) / `1024`(wukong) | 批次大小 |
| `--warmup-rounds` | `5` | 预热轮数 |
| `--inference-rounds` | `20` | 推理轮数 |
| `--musa-plugin` | 自动探测 `libmusa_plugin.so` | MUSA插件路径，可被 `MUSA_PLUGIN_PATH` 或命令行覆盖 |

## `.so` 加载入口与路径策略

当前 `tf_test_model` 中会主动加载 `libmusa_plugin.so` 的脚本有：

- `prunedGraph/run_graph_tf_musa.py`
- `wukong/run_wukong_tf_musa.py`
- `wukong/test_tf_musa_extension.py`

这些脚本统一通过 `utils.py` 里的路径解析逻辑加载插件。

默认优先级：

1. `MUSA_PLUGIN_PATH` 环境变量
2. 相邻工作区相对路径 `../tensorflow_musa_extension/build/libmusa_plugin.so`
3. 相邻工作区相对路径 `../tensorflow_musa_extension/build_local/libmusa_plugin.so`
4. Docker 绝对路径 `/workspace/tensorflow_musa_extension/build/libmusa_plugin.so`

推荐优先使用相邻工作区路径；Docker 绝对路径保留为容器内 fallback。

如果需要手动填写路径，推荐直接覆盖 `--musa-plugin` 参数：

```bash
# 推荐：相邻工作区相对路径
python prunedGraph/run_graph_tf_musa.py --device musa \
  --musa-plugin ../tensorflow_musa_extension/build/libmusa_plugin.so

# 容器内常见绝对路径
python prunedGraph/run_graph_tf_musa.py --device musa \
  --musa-plugin /workspace/tensorflow_musa_extension/build/libmusa_plugin.so
```

`wukong/test_tf_musa_extension.py` 也支持两种用法：

```bash
# 自动探测默认路径
python3 wukong/test_tf_musa_extension.py

# 手动指定路径
python3 wukong/test_tf_musa_extension.py ../tensorflow_musa_extension/build/libmusa_plugin.so
```

## 运行模式

### 1. 整网推理性能测试 (推荐)

```bash
# prunedGraph模型
python prunedGraph/run_graph_tf_musa.py --inference-only

# wukong模型  
python wukong/run_wukong_tf_musa.py --inference-only

# Wukong 精度对比
python wukong/run_wukong_tf_musa.py --compare-accuracy

# 指定自定义插件路径
python wukong/run_wukong_tf_musa.py --compare-accuracy \
  --musa-plugin /workspace/tensorflow_musa_extension/build/libmusa_plugin.so

# 测试 libmusa_plugin.so
python3 wukong/test_tf_musa_extension.py /path/to/libmusa_plugin.so
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

### 融合后算子可见性说明

- 在开启 `MUSA_DUMP_GRAPHDEF=1` 后，`prunedGraph` 和 `wukong` 的 profiler 都会优先读取最新的 `after_fusion.pbtxt`，用优化后图覆盖 profiler 中的节点类型映射。
- 这样做的目的是让 profiler 汇总尽量反映优化后的真实算子类型，而不是原始图中的基础算子类型。
- 最终能看到哪些融合算子，取决于具体模型结构和图优化是否命中对应 pattern；不同模型看到的融合算子可能不同。

示例：

```bash
export MUSA_DUMP_GRAPHDEF=1
export MUSA_DUMP_GRAPHDEF_DIR=/workspace/tensorflow_musa_extension/graph_debug
python prunedGraph/run_graph_tf_musa.py --profile-ops
```

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
