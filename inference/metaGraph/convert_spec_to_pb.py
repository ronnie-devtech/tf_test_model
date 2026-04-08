#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def random_array(shape, np_dtype, rng):
    if shape is None:
        shape = []
    shape = list(shape)
    if np_dtype in (np.str_, np.object_, np.bytes_):
        total = int(np.prod(shape)) if shape else 1
        values = np.array([f"s{rng.integers(0, 1_000_000)}".encode("utf-8") for _ in range(total)], dtype=object)
        return values.reshape(shape) if shape else values.reshape(()).item()
    if np.issubdtype(np_dtype, np.floating):
        return rng.standard_normal(size=shape).astype(np_dtype)
    if np.issubdtype(np_dtype, np.complexfloating):
        real = rng.standard_normal(size=shape)
        imag = rng.standard_normal(size=shape)
        return (real + 1j * imag).astype(np_dtype)
    if np.issubdtype(np_dtype, np.integer):
        # Integer variables are often used as indices/steps; keep them safe.
        return np.zeros(shape=shape, dtype=np_dtype)
    if np.issubdtype(np_dtype, np.bool_):
        return np.zeros(shape=shape, dtype=np.bool_)
    raise TypeError(f"Unsupported dtype for random init: {np_dtype}")


def load_meta_graph(spec_path: Path):
    meta = tf.compat.v1.MetaGraphDef()
    meta.ParseFromString(spec_path.read_bytes())
    if not meta.graph_def.node:
        raise ValueError(f"Spec does not contain graph_def nodes: {spec_path}")
    return meta


def tensor_names_from_collection(meta, key):
    coll = meta.collection_def.get(key)
    if not coll:
        return []
    if coll.WhichOneof("kind") != "node_list":
        raise ValueError(f"collection_def['{key}'] is not node_list")
    return list(coll.node_list.value)


def annotate_names_with_op(names, op_map):
    out = []
    for name in names:
        node_name = name.split(":")[0]
        out.append(f"{node_name}:{op_map.get(node_name, 'UnknownOp')}")
    return out


def node_diff(orig_graph_def, frozen_graph_def):
    orig_nodes = list(orig_graph_def.node)
    frozen_nodes = list(frozen_graph_def.node)
    orig_names = {n.name for n in orig_nodes}
    frozen_names = {n.name for n in frozen_nodes}

    orig_op_map = {n.name: n.op for n in orig_nodes}
    frozen_op_map = {n.name: n.op for n in frozen_nodes}
    return {
        "orig_node_count": len(orig_nodes),
        "pb_node_count": len(frozen_nodes),
        "removed_in_pb": annotate_names_with_op(sorted(orig_names - frozen_names), orig_op_map),
        "added_in_pb": annotate_names_with_op(sorted(frozen_names - orig_names), frozen_op_map),
        "orig_op_histogram": dict(sorted(Counter(n.op for n in orig_nodes).items())),
        "pb_op_histogram": dict(sorted(Counter(n.op for n in frozen_nodes).items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert MetaGraph spec to frozen PB with random variable init.")
    parser.add_argument("--spec", required=True, help="Path to *.spec (MetaGraphDef).")
    parser.add_argument("--out_root", default="frozen_out", help="Output root directory.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for variable init.")
    parser.add_argument("--no_copy_spec", default=True, help="Do not copy original spec into output dir.")
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)

    # stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # run_dir = Path(args.out_root).resolve() / f"{stamp}_{spec_path.stem}"
    run_dir = Path(args.out_root).resolve() / f"{spec_path.stem}"

    run_dir.mkdir(parents=True, exist_ok=False)
    m = re.search(r"(\d+)$", spec_path.stem)
    spec_id = m.group(1) if m else spec_path.stem

    meta = load_meta_graph(spec_path)
    input_spec = tensor_names_from_collection(meta, "input_spec")
    output_spec = tensor_names_from_collection(meta, "output_spec")
    
    output_nodes = sorted({name.split(":")[0] for name in output_spec})
    if not output_nodes:
        raise ValueError("output_spec is empty, cannot freeze graph.")

    rng = np.random.default_rng(args.seed)
    with tf.Graph().as_default():
        tf.compat.v1.train.import_meta_graph(meta_graph_or_file=meta, clear_devices=True)
        graph = tf.compat.v1.get_default_graph()
        vars_all = list({v.ref(): v for v in (tf.compat.v1.global_variables() + tf.compat.v1.local_variables())}.values())
        assign_ops = []
        for var in vars_all:
            shape = var.shape.as_list()
            if any(dim is None for dim in shape):
                raise ValueError(f"Variable has unknown shape, cannot random-init safely: {var.name} shape={shape}")
            np_dtype = var.dtype.as_numpy_dtype
            value = random_array(shape, np_dtype, rng)
            assign_ops.append(tf.compat.v1.assign(var, value, validate_shape=True))

        with tf.compat.v1.Session(graph=graph) as sess:
            if assign_ops:
                sess.run(assign_ops)
            frozen = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=output_nodes,
            )

    pb_path = run_dir / f"frozen_graph_{spec_id}.pb"
    pb_path.write_bytes(frozen.SerializeToString())

    diff = node_diff(meta.graph_def, frozen)
    orig_op_map = {n.name: n.op for n in meta.graph_def.node}
    frozen_op_map = {n.name: n.op for n in frozen.node}
    manifest = {
        "spec_path": str(spec_path),
        "pb_path": str(pb_path),
        "seed": args.seed,
        "input_spec_count": len(input_spec),
        "output_spec_count": len(output_spec),
        "input_spec": annotate_names_with_op(input_spec, orig_op_map),
        "output_spec": annotate_names_with_op(output_spec, orig_op_map),
        "freeze_output_nodes": annotate_names_with_op(output_nodes, frozen_op_map),
        "node_diff_file": str(run_dir / "node_diff.json"),
        "removed_in_pb": diff["removed_in_pb"],
    }

    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "node_diff.json").write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_copy_spec:
        shutil.copy2(spec_path, run_dir / spec_path.name)

    print(f"[OK] output_dir={run_dir}")
    print(f"[OK] pb={pb_path}")
    print(f"[OK] node_diff={run_dir / 'node_diff.json'}")


if __name__ == "__main__":
    main()


'''
用法： python convert_spec_to_pb.py --spec /path/to/*.spec
'''