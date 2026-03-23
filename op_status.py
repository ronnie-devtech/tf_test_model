#!/usr/bin/env python3
"""将 op_stats_MUSA_xxx.json 转换为 CSV 文件。"""

import argparse
import json
import csv


def main():
    parser = argparse.ArgumentParser(description="将 op_stats_MUSA_xxx.json 转换为 CSV 文件")
    parser.add_argument("--input", required=True, help="输入的 JSON 文件路径")
    parser.add_argument("--output", required=True, help="输出的 CSV 文件路径")
    args = parser.parse_args()

    # 读取 JSON 文件
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 写入 CSV 文件
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(["name", "device", "duration_ms", "op_type"])
        # 写入数据行
        for op in data.get("operators", []):
            writer.writerow(
                [
                    op.get("name", ""),
                    op.get("device", ""),
                    op.get("duration_ms", ""),
                    op.get("op_type", ""),
                ]
            )

    print(f"转换完成：{args.input} -> {args.output}")
    print(f"共 {len(data.get('operators', []))} 条算子记录")


if __name__ == "__main__":
    main()
