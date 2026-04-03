# TensorFlow MUSA 模型训练测试

本目录包含多个推荐模型的 TensorFlow MUSA 扩展训练测试。

## 目录结构

```
training/
├── run_all_training_tests.py  # 统一测试入口脚本
├── test_utils.py              # 公共测试工具函数
├── deepfm/                    # DeepFM 模型
├── dien/                      # DIEN 模型
├── din/                       # DIN 模型
├── dsin/                      # DSIN 模型
├── esmm/                      # ESMM 模型
├── fgcnn/                     # FGCNN 模型
├── flen/                      # FLEN 模型
├── fwfm/                      # FwFM 模型
├── mmoe/                      # MMoE 模型
├── onetrans/                  # OneTrans 模型
├── ple/                       # PLE 模型
├── rankmixer/                 # RankMixer 模型
├── tokenmixer-large/          # TokenMixer-Large 模型
├── wukong/                    # WuKong 模型（训练）
└── xdeepfm/                   # xDeepFM 模型

inference/
├── prunedGraph/               # 腾讯中台模型
└── wukong/                    # WuKong 模型（推理）
```

## 训练样例使用方法
```bash
cd training
```
### 查看可用模型

```bash
python run_all_training_tests.py --list-models
```

### 运行所有模型测试

```bash
python run_all_training_tests.py --epochs 10 --musa-plugin ../../tensorflow_musa_extension/build/libmusa_plugin.so
```

### 运行指定模型测试

```bash
python run_all_training_tests.py --epochs 10 --models deepfm wukong dien --musa-plugin ../../tensorflow_musa_extension/build/libmusa_plugin.so
```

### 指定 GPU 设备

```bash
# 单 GPU
python run_all_training_tests.py --epochs 10 --gpu 0 --musa-plugin ../../tensorflow_musa_extension/build/libmusa_plugin.so

# 多 GPU
python run_all_training_tests.py --epochs 10 --gpu 0,1,2 --musa-plugin ../../tensorflow_musa_extension/build/libmusa_plugin.so
```

### 指定错误日志目录

```bash
python run_all_training_tests.py --epochs 10 --log-dir my_logs --musa-plugin ../../tensorflow_musa_extension/build/libmusa_plugin.so
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 10 |
| `--musa-plugin` | TensorFlow MUSA 库 .so 文件路径 | 无 |
| `--models` | 指定要测试的模型列表 | 所有模型 |
| `--gpu` | GPU 设备 ID (如: 0 或 0,1,2) | 无 |
| `--log-dir` | 错误日志目录名 | error_logs |
| `--list-models` | 列出所有可测试的模型 | - |

### 输出说明

测试完成后会输出结果摘要：

```
[OK] deepfm: PASSED
[OK] wukong: PASSED
[FAIL] dien: FAILED

总计: 15 个模型
成功: 14
失败: 1
```

失败的模型错误详情会保存在 `--log-dir` 目录下的 `<model_name>_error.log` 文件中。

### 单独测试某个模型

也可以直接进入模型目录运行单独测试：

```bash
cd deepfm
python test_tf_musa_extension.py --musa_plugin ../../tensorflow_musa_extension/build/libmusa_plugin.so --epochs 10
```

## 推理样例使用方法
```bash
cd inference
```
进入对应模型目录并运行推理代码即可
