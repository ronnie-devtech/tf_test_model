# Standard Model Performance Testing Framework

This directory contains standard-model performance testing scripts used to evaluate end-to-end behavior and performance of the TensorFlow MUSA PluggableDevice.

## Directory Layout

```text
standard_model/
├── prunedGraph/               # Pruned graph model (GraphDef format)
│   ├── run_graph_tf_musa.py   # Main test entry
│   └── logs/                  # Logs and trace outputs
├── wukong/                    # Wukong deep learning model
│   ├── run_wukong_tf_musa.py
│   ├── test_tf_musa_extension.py
│   └── logs/
├── rankmixer/
│   ├── test_tf_musa_extension.py
│   └── logs/
├── onetrans/
│   ├── test_tf_musa_extension.py
│   └── logs/
├── tokenmixer-large/
│   ├── test_tf_musa_extension.py
│   └── logs/
├── fgcnn/
│   ├── test_tf_musa_extension.py
│   └── logs/
└── fwfm/
    ├── test_tf_musa_extension.py
    └── logs/
```

## Test Model Description

### 1. prunedGraph

`prunedGraph` is a fixed `GraphDef (.pb)` model and is well suited for validating:

- whether the custom graph optimizer is loaded correctly
- whether fusion patterns are matched
- whether fused custom operators really enter the final execution graph
- whether end-to-end CPU vs MUSA accuracy is consistent

This model is especially useful for graph-optimization development because it is naturally graph-based and makes it easy to:

- dump `before_fusion / after_fusion / final` graphs
- inspect subgraph replacement directly in Netron
- verify whether fused operators remain reachable on the final output path

Based on the current model and logs, common operator types in `prunedGraph` include:

- `MatMul` / `BatchMatMulV2`
- `BiasAdd`
- `AddV2` / `Mul` / `Sub` / `RealDiv`
- `Reshape` / `Transpose` / `Pack` / `ConcatV2`
- `GatherV2`
- `Mean` / `Prod` / `Select`
- `FusedBatchNormV3`
- `Sqrt` / `Rsqrt`
- fused operators introduced by graph optimization, such as `MusaGelu`

If the goal is graph fusion development, `prunedGraph` is the recommended first validation entry.

### 2. wukong / rankmixer / onetrans / tokenmixer-large / fwfm / fgcnn / xdeepfm / dien / dsin

These directories contain TensorFlow/Keras dynamic models. As a group, they are more suitable for:

- operator pass-rate validation on MUSA
- end-to-end forward execution validation after plugin loading
- validating eager and `tf.function` execution paths
- end-to-end inference performance and stability checks

Compared with `prunedGraph`, they are closer to model-level functional and performance validation than strict GraphDef fusion visualization.

From the model structures, this group can be roughly divided into three categories:

- `wukong` / `rankmixer` / `onetrans` / `tokenmixer-large`
  These models focus more on embedding, token mixing, attention, LayerNorm, and MLP-style blocks, and are useful for validating end-to-end execution of `MatMul`, `BatchMatMulV2`, `Softmax`, `LayerNormalization`, `Reshape`, `Transpose`, `ConcatV2`, and `Stack`.
- `fwfm` / `fgcnn` / `xdeepfm`
  These CTR-style models focus on feature interaction. In addition to embedding and MLP blocks, they also cover inner-product, convolution, flatten, and feature recombination paths, making them useful for validating `Conv1D`, `Conv2D`, `MatMul`, `BatchNormalization`, `Relu`, `ConcatV2`, and `Reshape`.
- `dien` / `dsin`
  These sequential recommendation models contain stronger sequence modeling and interest-evolution structures, covering attention, GRU/LSTM, bidirectional sequence structures, and softmax-based scoring paths.

Common operator types covered by this model group include:

- `ResourceGather` / `ReadVariableOp`
- `MatMul` / `BatchMatMulV2` / `Einsum`
- `FusedBatchNormV3` / `LayerNormalization`
- `Relu` / `Gelu` / `Sigmoid` / `Tanh` / `Softmax`
- `BiasAdd` / `AddV2` / `Mul` / `Sub`
- `Reshape` / `Transpose` / `StridedSlice` / `Pack` / `Stack` / `Split` / `ExpandDims` / `Squeeze` / `Tile` / `Fill`
- `ConcatV2` / `MatrixSetDiag`
- `ReduceMean` / `ReduceSum`
- `Conv1D` / `Conv2D` / `Flatten`
- `LSTM` / `GRU`
- `Rsqrt` / `Sqrt` / `Square` / `Pow`
- `Cast` / `Equal` / `Where` / `ClipByValue`
- `TensorArray` / `While` / `SequenceMask`
- `Send` (cross-device transfer events visible in profiler traces)

So this model group is better used for end-to-end functionality, operator pass-rate, device execution, and performance validation, rather than as the primary signal for whether a specific graph-fusion pattern is hit.

| Model | Pass Rate |
|---|---|
| wukong | ✅ |
| rankmixer | ✅ |
| tokenmixer-large | ✅ |
| fwfm | ✅ |
| xdeepfm | ✅ |
| dsin | ✅ |
| fgcnn | ✅ |
| onetrans | ❌ |
| dien | ❌ |

## Common Command-Line Arguments

The model scripts support the following common arguments:

| Argument | Default | Description |
|---|---|---|
| `--device` | `musa` | Target device: `cpu` or `musa` |
| `--batch-size` | `100` (`prunedGraph`) / `1024` (`wukong`) | Batch size |
| `--warmup-rounds` | `5` | Warmup iterations |
| `--inference-rounds` | `20` | Inference iterations |
| `--musa-plugin` | auto-detect `libmusa_plugin.so` | Plugin path, overridable by `MUSA_PLUGIN_PATH` or CLI argument |

## `.so` Loading Entry and Path Strategy

The scripts in `tf_test_model` that actively load `libmusa_plugin.so` are:

- `prunedGraph/run_graph_tf_musa.py`
- `wukong/run_wukong_tf_musa.py`
- `wukong/test_tf_musa_extension.py`

These scripts all rely on the shared path-resolution logic in `utils.py`.

Default priority:

1. `MUSA_PLUGIN_PATH` environment variable
2. Relative sibling-workspace path `../tensorflow_musa_extension/build/libmusa_plugin.so`
3. Relative sibling-workspace path `../tensorflow_musa_extension/build_local/libmusa_plugin.so`
4. Docker absolute path `/workspace/tensorflow_musa_extension/build/libmusa_plugin.so`

Using the sibling-workspace path is recommended. The Docker absolute path is kept as an in-container fallback.

If you want to specify the plugin manually, it is recommended to override `--musa-plugin` directly:

```bash
# Recommended: sibling-workspace relative path
python prunedGraph/run_graph_tf_musa.py --device musa \
  --musa-plugin ../tensorflow_musa_extension/build/libmusa_plugin.so

# Common absolute path inside containers
python prunedGraph/run_graph_tf_musa.py --device musa \
  --musa-plugin /workspace/tensorflow_musa_extension/build/libmusa_plugin.so
```

`wukong/test_tf_musa_extension.py` also supports two modes:

```bash
# Auto-detect the default path
python3 wukong/test_tf_musa_extension.py

# Manually specify the plugin path
python3 wukong/test_tf_musa_extension.py ../tensorflow_musa_extension/build/libmusa_plugin.so
```

## Run Modes

### 1. End-to-End Inference Performance Test (Recommended)

```bash
# prunedGraph
python prunedGraph/run_graph_tf_musa.py --inference-only

# wukong
python wukong/run_wukong_tf_musa.py --inference-only

# Wukong accuracy comparison
python wukong/run_wukong_tf_musa.py --compare-accuracy

# Specify a custom plugin path
python wukong/run_wukong_tf_musa.py --compare-accuracy \
  --musa-plugin /workspace/tensorflow_musa_extension/build/libmusa_plugin.so

# Test libmusa_plugin.so directly
python3 wukong/test_tf_musa_extension.py /path/to/libmusa_plugin.so
```

Output locations:

- `prunedGraph/logs/graph_inference/YYYY-MM-DD-HH.MM.SS_trace/`
- `wukong/logs/tensorflow_inference/YYYY-MM-DD-HH.MM.SS_trace/`

Generated files:

- `inference_only_result_MUSA_*.json` - performance results
- `inference_only_result_CPU_*.json` - CPU comparison results

### 2. Operator-Level Performance Analysis

```bash
# prunedGraph
python prunedGraph/run_graph_tf_musa.py --profile-ops

# wukong
python wukong/run_wukong_tf_musa.py --profile-ops
```

Output location: same as above, with additional files:

- `ops_profile_MUSA_*/` - TensorFlow Profiler traces
- `operator_timings_*.json` - operator execution timing statistics

### Fused-Operator Visibility Notes

- When `MUSA_DUMP_GRAPHDEF=1` is enabled, both `prunedGraph` and `wukong` profilers will prefer the latest `after_fusion.pbtxt` and use the optimized graph to override node-type mapping in profiler output.
- This is done so that the profiler summary reflects optimized operator types as much as possible, instead of the original unfused base operators.
- Which fused operators become visible ultimately depends on the actual model structure and whether the graph optimizer hits the corresponding pattern. Different models may expose different fused operators.

Example:

```bash
export MUSA_DUMP_GRAPHDEF=1
export MUSA_DUMP_GRAPHDEF_DIR=/workspace/tensorflow_musa_extension/graph_debug
python prunedGraph/run_graph_tf_musa.py --profile-ops
```

### 3. Full Analysis Mode

```bash
# Run the full analysis with default settings
python prunedGraph/run_graph_tf_musa.py
python wukong/run_wukong_tf_musa.py
```

Includes:

- end-to-end performance analysis
- operator-level performance analysis
- model-structure analysis (`wukong`)
- device information analysis

## Performance Comparison Suggestions

1. CPU vs MUSA comparison: run the same configuration under both `--device cpu` and `--device musa`
2. Different batch sizes: test performance impact across multiple batch sizes
3. Average over multiple runs: because system load may fluctuate, using multiple runs is recommended

## Output File Description

### Performance Result File (`inference_only_result_*.json`)

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

### Operator Timing File (`operator_timings_*.json`)

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

## Troubleshooting

1. Failed to load the MUSA plugin: make sure `tensorflow_musa_extension` has been built correctly
2. Model file not found: make sure the model file is in the expected location or provide `--model`
3. Out of memory: reduce `--batch-size`
4. Abnormal performance: check whether some operators fell back to CPU execution
