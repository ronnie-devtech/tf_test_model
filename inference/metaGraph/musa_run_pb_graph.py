#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
import json
import os
import re
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional, Union

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.config.run_functions_eagerly(True)

# ==========================================
# 全局配置
# ==========================================
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MUSA_PLUGIN_PATH = (
    SCRIPT_DIR / ".." / "tensorflow_musa_extension" / "build" / "libmusa_plugin.so"
).resolve()
GRAPH_DUMP_ENV = "MUSA_DUMP_GRAPHDEF"
GRAPH_DUMP_DIR_ENV = "MUSA_DUMP_GRAPHDEF_DIR"
GRAPH_DUMP_STAGE_SUFFIXES = {
    "after_default_grappler_before_musa_optimizer": "initial",
    "after_custom_fusion_pattern": "after_fusion",
    "final_after_all_musa_passes": "final",
}

# ==========================================
# 1. 加载 MUSA 插件
# ==========================================
def load_musa_plugin(plugin_path: Path):
    if plugin_path.exists():
        try:
            tf.load_op_library(str(plugin_path))
            print(f">>>> [MUSA] Plugin loaded successfully from: {plugin_path}")
            return True
        except Exception as e:
            print(f"!!!! [MUSA] Failed to load plugin: {e}")
    else:
        print(f"[Error] MUSA Plugin({plugin_path}) not found.")
    return False

def env_flag_enabled(name):
    value = os.environ.get(name, "")
    return value in ("1", "true", "TRUE", "yes")

def create_musa_dump_session_config(enable_musa_optimizer: bool, allow_soft_placement: bool, log_device_placement: bool):
    if not enable_musa_optimizer:
        return None

    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement = allow_soft_placement
    config.log_device_placement = log_device_placement

    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.min_graph_nodes = -1
    custom_optimizer = rewrite_options.custom_optimizers.add()
    custom_optimizer.name = "musa_graph_optimizer"
    rewrite_options.optimizers.extend(["musa_graph_optimizer"])
    return config

def collect_graph_dump_files(dump_dir: Union[str, Path, None]):
    if not dump_dir:
        return {}

    dump_root = Path(dump_dir)
    if not dump_root.exists():
        return {}

    files = {}
    for alias, stage_suffix in GRAPH_DUMP_STAGE_SUFFIXES.items():
        matches = []
        for ext in (".pbtxt", ".pb"):
            matches.extend(dump_root.glob(f"*_{stage_suffix}{ext}"))
        matches = sorted(matches, key=lambda path: (path.stat().st_mtime, str(path)))
        if matches:
            latest = matches[-1]
            files[alias] = {
                "stage": stage_suffix,
                "format": latest.suffix.lstrip("."),
                "path": str(latest.resolve()),
            }
    return files

@contextmanager
def configured_graph_dump_dir(default_dump_dir: Union[Path, None]):
    if not env_flag_enabled(GRAPH_DUMP_ENV):
        yield None
        return

    old_dump_dir = os.environ.get(GRAPH_DUMP_DIR_ENV)
    if old_dump_dir:
        dump_dir = Path(old_dump_dir).resolve()
    elif default_dump_dir is not None:
        dump_dir = Path(default_dump_dir).resolve()
        os.environ[GRAPH_DUMP_DIR_ENV] = str(dump_dir)
    else:
        dump_dir = Path.cwd().resolve()

    dump_dir.mkdir(parents=True, exist_ok=True)

    try:
        yield dump_dir
    finally:
        if old_dump_dir is None:
            os.environ.pop(GRAPH_DUMP_DIR_ENV, None)


def load_meta(spec_path: Path):
    meta = tf.compat.v1.MetaGraphDef()
    meta.ParseFromString(spec_path.read_bytes())
    if not meta.graph_def.node:
        raise ValueError(f"Invalid spec, graph_def is empty: {spec_path}")
    return meta


def read_node_list_collection(meta, key):
    coll = meta.collection_def.get(key)
    if not coll:
        raise ValueError(f"Missing collection_def['{key}'] in spec.")
    if coll.WhichOneof("kind") != "node_list":
        raise ValueError(f"collection_def['{key}'] is not node_list.")
    return list(coll.node_list.value)


def _shape_from_node_attr(node):
    if "_output_shapes" in node.attr and node.attr["_output_shapes"].list.shape:
        shp = node.attr["_output_shapes"].list.shape[0]
        return [d.size if d.size != -1 else None for d in shp.dim]
    if "shape" in node.attr:
        shp = node.attr["shape"].shape
        return [d.size if d.size != -1 else None for d in shp.dim]
    return None


def build_spec_tensor_shape_map(meta):
    out = {}
    for n in meta.graph_def.node:
        shp = _shape_from_node_attr(n)
        if shp is not None:
            out[f"{n.name}:0"] = shp
    return out


def merge_shape(spec_shape, pb_shape):
    if spec_shape is None:
        return list(pb_shape) if pb_shape is not None else []
    if pb_shape is None:
        return list(spec_shape)
    n = max(len(spec_shape), len(pb_shape))
    merged = []
    for i in range(n):
        s = spec_shape[i] if i < len(spec_shape) else None
        p = pb_shape[i] if i < len(pb_shape) else None
        merged.append(s if s is not None else p)
    return merged


def resolve_shape(shape, bs, unknown_dim):
    if shape is None:
        return []
    out = []
    for i, dim in enumerate(shape):
        if dim is None:
            if i:
                raise ValueError("spec has unknown_dim in non-bs")
            out.append(bs if i == 0 else unknown_dim)
        else:
            out.append(dim)
    return out


def parse_bs_values(bs_arg):
    if isinstance(bs_arg, int):
        return [bs_arg]
    parts = [x.strip() for x in str(bs_arg).split(",") if x.strip()]
    if not parts:
        raise ValueError("bs is empty")
    values = []
    seen = set()
    for p in parts:
        v = int(p)
        if v <= 0:
            raise ValueError(f"batch size must be > 0, got {v}")
        if v not in seen:
            seen.add(v)
            values.append(v)
    return values


def parse_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {v}")


def random_array(shape, np_dtype, rng):
    if np_dtype in (np.str_, np.object_, np.bytes_, object):
        total = int(np.prod(shape)) if shape else 1
        vals = np.array([f"s{rng.integers(0, 1_000_000)}".encode("utf-8") for _ in range(total)], dtype=object)
        return vals.reshape(shape) if shape else vals.reshape(()).item()
    if np.issubdtype(np_dtype, np.floating):
        return rng.uniform(0.1, 1.0, size=shape).astype(np_dtype)
    if np.issubdtype(np_dtype, np.complexfloating):
        real = rng.standard_normal(size=shape)
        imag = rng.standard_normal(size=shape)
        return (real + 1j * imag).astype(np_dtype)
    if np.issubdtype(np_dtype, np.integer):
        return rng.integers(0, 10, size=shape, dtype=np_dtype)
    if np.issubdtype(np_dtype, np.bool_):
        return rng.choice([False, True], size=shape)
    raise TypeError(f"Unsupported dtype for random input: {np_dtype}")


def percentile(arr, q):
    if not arr:
        return 0.0
    return float(np.percentile(np.array(arr, dtype=np.float64), q))


def parse_spec_id(spec_path: Path):
    m = re.search(r"(\d+)$", spec_path.stem)
    return m.group(1) if m else spec_path.stem


def detect_pb(spec_path: Path, explicit_pb: Union[str, None], extra_search_roots=None):
    if explicit_pb:
        pb = Path(explicit_pb).resolve()
        if not pb.exists():
            raise FileNotFoundError(pb)
        return pb

    spec_id = parse_spec_id(spec_path)
    pb_name = f"frozen_graph_{spec_id}.pb"
    search_roots = [
        spec_path.parent,
        Path.cwd(),
        Path.cwd() / "artifacts",
        Path.cwd() / "frozen_out",
    ]
    if extra_search_roots:
        search_roots.extend(extra_search_roots)

    candidates = []
    seen = set()
    for root in search_roots:
        root = Path(root).resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        candidates.extend(root.glob(f"**/{pb_name}"))
    if not candidates:
        raise FileNotFoundError(f"Auto-detect pb failed, expected file name: {pb_name}")
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def _strip_tensor_name(name):
    base = name[1:] if name.startswith("^") else name
    return base.split(":")[0]


def _const_int_list(node):
    if node.op != "Const" or "value" not in node.attr:
        return None
    t = node.attr["value"].tensor
    try:
        arr = tf.make_ndarray(t)
        if np.issubdtype(arr.dtype, np.integer):
            return [int(x) for x in arr.reshape(-1).tolist()]
    except Exception:
        pass
    if t.int_val:
        return list(t.int_val)
    if t.int64_val:
        return [int(x) for x in t.int64_val]
    return None


def infer_placeholder_min_dims(meta):
    node_map = {n.name: n for n in meta.graph_def.node}
    mins = {}
    for n in meta.graph_def.node:
        if n.op != "Slice" or len(n.input) < 3:
            continue
        x_name = _strip_tensor_name(n.input[0])
        x_node = node_map.get(x_name)
        if not x_node or x_node.op not in ("Placeholder", "PlaceholderWithDefault"):
            continue
        b = node_map.get(_strip_tensor_name(n.input[1]))
        s = node_map.get(_strip_tensor_name(n.input[2]))
        begin = _const_int_list(b) if b else None
        size = _const_int_list(s) if s else None
        if not begin or not size:
            continue
        tname = f"{x_name}:0"
        req = mins.setdefault(tname, {})
        for i, (bi, si) in enumerate(zip(begin, size)):
            if i == 0:
                continue
            if bi < 0:
                continue
            need = bi + (si if si > 0 else 1)
            req[i] = max(req.get(i, 0), int(need))
    return mins


def extract_core_error(stack: Union[str, None]):
    if not stack:
        return None
    lines = [ln.strip() for ln in stack.strip().splitlines() if ln.strip()]
    key_patterns = (
        "ResourceExhaustedError",
        "InvalidArgumentError",
        "NotFoundError",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "ran out of memory",
        "oom",
    )
    for ln in reversed(lines):
        low = ln.lower()
        if "original stack trace" in low:
            continue
        if any(k.lower() in low for k in key_patterns):
            return ln
    for ln in reversed(lines):
        if ln.startswith("File "):
            continue
        if ln.startswith("Traceback"):
            continue
        if "original stack trace" in ln.lower():
            continue
        return ln
    return lines[-1] if lines else None


def run_single_spec(spec_path: Path, pb_path: Path, args, bs: int, musa_loaded: bool, runner_out: Path):
    meta = load_meta(spec_path)
    input_spec = read_node_list_collection(meta, "input_spec")
    output_spec = read_node_list_collection(meta, "output_spec")
    spec_shape_map = build_spec_tensor_shape_map(meta)
    slice_min_dims = infer_placeholder_min_dims(meta)

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(pb_path.read_bytes())

    with tf.Graph().as_default() as graph:
        if args.device:
            with tf.device(args.device):
                tf.import_graph_def(graph_def, name="")
        else:
            tf.import_graph_def(graph_def, name="")
        outputs = [graph.get_tensor_by_name(name) for name in output_spec]
        def safe_shape(tensor):
            if tensor.shape.rank is None:
                return None
            return [d if d is not None else None for d in tensor.shape.as_list()]
        output_info = [{"name": t.name, "optype": t.op.type, "dtype": t.dtype.name, "shape_in_graph": safe_shape(t)} for t in outputs]

        run_error = None
        lat_ms = []
        last_vals = None
        rng = np.random.default_rng(args.seed)
        inputs = []
        feed_dict = {}
        for name in input_spec:
            tensor = graph.get_tensor_by_name(name)
            tensor_shape = safe_shape(tensor)
            spec_shape = spec_shape_map.get(name)
            merged_shape = merge_shape(spec_shape, tensor_shape)
            if name in slice_min_dims:
                req = slice_min_dims[name]
                merged_shape = list(merged_shape)
                for dim_idx, min_dim in req.items():
                    if dim_idx >= len(merged_shape):
                        continue
                    cur = merged_shape[dim_idx]
                    if cur is None or cur < min_dim:
                        merged_shape[dim_idx] = min_dim
            run_shape = resolve_shape(merged_shape, bs, args.unknown_dim)
            if tensor.op.type in ("Placeholder", "PlaceholderWithDefault"):
                value = random_array(run_shape, tensor.dtype.as_numpy_dtype, rng)
                feed_dict[tensor] = value
        ### musa dump
        graph_dump = {
            "enabled": env_flag_enabled(GRAPH_DUMP_ENV),
            "plugin_loaded": musa_loaded,
            "optimizer_enabled": False,
            "dump_dir": None,
            "files": {},
        }

        enable_musa_optimizer = musa_loaded
        default_dump_dir = None
        session_config = None
        
        if enable_musa_optimizer:   # dump
            session_config = create_musa_dump_session_config(enable_musa_optimizer=True, allow_soft_placement=bool(args.allow_soft_placement), log_device_placement=bool(args.log_device_placement))
            graph_dump["optimizer_enabled"] = True
            if graph_dump["enabled"]:
                default_dump_dir = (runner_out / f"{spec_path.stem}_bs_{bs}").resolve()

        with configured_graph_dump_dir(default_dump_dir) as active_dump_dir:
            if active_dump_dir is not None:
                graph_dump["dump_dir"] = str(active_dump_dir)    
            with tf.compat.v1.Session(graph=graph, config=session_config) as sess:
                try:
                    for _ in range(max(0, args.warmup)):
                        sess.run(outputs, feed_dict=feed_dict)
                    for _ in range(max(1, args.run_iters)):
                        t0 = time.perf_counter()
                        last_vals = sess.run(outputs, feed_dict=feed_dict)
                        t1 = time.perf_counter()
                        lat_ms.append((t1 - t0) * 1000.0)
                except Exception:
                    run_error = traceback.format_exc()

        if graph_dump["dump_dir"]:
            graph_dump["files"] = collect_graph_dump_files(graph_dump["dump_dir"])

    return {
        "spec_path": str(spec_path),
        "pb_path": str(pb_path),
        "batch_size": bs,
        "num of outputs": len(output_info),
        "outputs": output_info,
        "status": "ok" if run_error is None else "failed",
        "error_core": extract_core_error(run_error),
        "error": run_error,
        "graph_dump": graph_dump,
        "timing_ms": {
            "warmup": max(0, args.warmup),
            "run_iters": max(1, args.run_iters),
            "average": float(np.mean(lat_ms)) if lat_ms else 0.0,
            "min": float(np.min(lat_ms)) if lat_ms else 0.0,
            "max": float(np.max(lat_ms)) if lat_ms else 0.0,
            "p50": percentile(lat_ms, 50),
            "p90": percentile(lat_ms, 90),
            "p95": percentile(lat_ms, 95),
            "all": lat_ms,
        },
    }


def collect_specs(spec: Union[str, None], spec_dir: Union[str, None]):
    if spec:
        p = Path(spec).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return [p]

    root = Path(spec_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    specs = sorted(root.rglob("*.spec"))
    if not specs:
        raise FileNotFoundError(f"No .spec files found under: {root}")
    return specs


def convert_spec_to_pb(spec_path: Path, convert_script: Path, seed: int):
    cmd = [
        sys.executable,
        str(convert_script),
        "--spec",
        str(spec_path),
        "--seed",
        str(seed),
    ]
    env = dict(os.environ)
    # Conversion only needs graph freezing; forcing CPU avoids GPU OOM contention
    # with the long-lived runner process.
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        core = extract_core_error(res.stderr) or extract_core_error(res.stdout)
        raise RuntimeError(
            f"convert failed for {spec_path}\n"
            f"core_error: {core}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{res.stdout}\n"
            f"stderr:\n{res.stderr}"
        )

    spec_id = parse_spec_id(spec_path)
    pb_path = Path("./frozen_out") / spec_path.stem / f"frozen_graph_{spec_id}.pb"
    if not pb_path.exists():
        raise FileNotFoundError(f"convert succeeded but pb missing: {pb_path}")
    return pb_path.resolve()


def main():
    parser = argparse.ArgumentParser(description="Run inference from frozen PB using I/O tensors from spec.")
    parser.add_argument("--spec", default=None, help="Path to a single *.spec (MetaGraphDef).")
    parser.add_argument("--spec_dir", default=None, help="Directory to recursively scan all *.spec files.")
    parser.add_argument("--pb", default=None, help="Path to single frozen_graph_*.pb. Only valid with --spec.")
    parser.add_argument("--bs", default="1024", help="Batch size for unresolved first dimension. Supports single value or comma-separated list, e.g. 1,2,4,8.")
    parser.add_argument("--unknown_dim", type=int, default=1, help="Fill value for unresolved non-batch dimensions.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations.")
    parser.add_argument("--run_iters", type=int, default=10, help="Measured iterations.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--out_root", default="runner_out", help="Output root directory.")
    parser.add_argument("--strict", type=parse_bool, default=True, help="Exit non-zero when any runtime/convert fails.")
    parser.add_argument("--device", default="/device:MUSA:0", choices=["/device:MUSA:0", "/device:CPU:0"], help="Device scope for imported graph, e.g. /device:MUSA:0 or /CPU:0.")
    parser.add_argument("--allow_soft_placement", type=parse_bool, default=True, help="Whether TensorFlow can place unsupported ops on other devices.")
    parser.add_argument("--log_device_placement", type=parse_bool, default=False, help="Print TensorFlow op placement logs.")
    parser.add_argument("--convert_script", default="convert_spec_to_pb.py", help="Path to convert spec->pb script.")
    parser.add_argument(
    "--musa-plugin",
    type=str,
    default=None,
    help="Path to MUSA plugin library",
    )

    args = parser.parse_args()

    if bool(args.spec) == bool(args.spec_dir):
        raise ValueError("Provide exactly one of --spec or --spec_dir")
    if args.pb and not args.spec:
        raise ValueError("--pb can only be used together with --spec")

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    bs_values = parse_bs_values(args.bs)
    convert_script = Path(args.convert_script).resolve()
    if not convert_script.exists():
        raise FileNotFoundError(f"convert script not found: {convert_script}")
    if args.device and "MUSA" in str(args.device).upper():
        musa_loaded = load_musa_plugin(Path(args.musa_plugin).resolve() if args.musa_plugin else DEFAULT_MUSA_PLUGIN_PATH)
        musa_devices = tf.config.list_physical_devices("MUSA")
        if not musa_devices:
            raise RuntimeError(
                f"requested device {args.device}, but no MUSA devices are visible"
            )
    else:
        musa_loaded = False
        print("[Info] MUSA Plugin loading skipped. Running on CPU.")

    specs = collect_specs(args.spec, args.spec_dir)
    auto_pb_root = out_root / datetime.now().strftime("%Y%m%d_%H%M%S")

    all_reports = []
    failures = 0
    for spec_path in specs:
        print(f"[INFO] processing spec: {spec_path}")

        pb_path = None
        detect_error = None
        try:
            pb_path = detect_pb(spec_path, args.pb, extra_search_roots=[out_root])
            print(f"[INFO] found pb: {pb_path}")
        except Exception as e:
            detect_error = e

        if pb_path is None:
            print(f"[INFO] pb missing for {spec_path.name}, auto converting by {convert_script.name} ...")
            try:
                pb_path = convert_spec_to_pb(spec_path, convert_script, args.seed)
                print(f"[INFO] auto-convert success: {pb_path}")
            except Exception:
                err = traceback.format_exc()
                failures += 1
                all_reports.append(
                    {
                        "spec_path": str(spec_path),
                        "pb_path": None,
                        "status": "failed",
                        "error_stage": "detect_or_convert_pb",
                        "error_core": extract_core_error(err) or str(detect_error),
                        "error": err,
                    }
                )
                continue

        try:
            for bs in bs_values:
                one_report = run_single_spec(spec_path.resolve(), pb_path.resolve(), args, bs, musa_loaded, auto_pb_root)
                if one_report["status"] != "ok":
                    failures += 1
                all_reports.append(one_report)
                print(f"[INFO] run done: spec={spec_path.name} bs={bs} status={one_report['status']}")
                if one_report.get("error_core"):
                    print(f"[INFO] core error: {one_report['error_core']}")
                dump_files = (one_report.get("graph_dump") or {}).get("files") or {}
                if dump_files:
                    for alias, info in dump_files.items():
                        print(f"[INFO] graph dump {alias}: {info['path']}")
        except Exception:
            err = traceback.format_exc()
            failures += 1
            for bs in bs_values:
                all_reports.append(
                    {
                        "spec_path": str(spec_path),
                        "pb_path": str(pb_path) if pb_path else None,
                        "batch_size": bs,
                        "status": "failed",
                        "error_stage": "run_inference",
                        "error_core": extract_core_error(err),
                        "error": err,
                    }
                )

    summary = {
        "total_specs": len(specs),
        "bs_values": bs_values,
        "total_runs": len(all_reports),
        "ok": sum(1 for x in all_reports if x.get("status") == "ok"),
        "failed": sum(1 for x in all_reports if x.get("status") != "ok"),
    }

    avg_time_summary = []
    latency_summary = []
    for r in all_reports:
        timing = r.get("timing_ms") or {}
        avg_time_summary.append(
            {
                "spec_path": r.get("spec_path"),
                "pb_path": r.get("pb_path"),
                "batch_size": r.get("batch_size"),
                "status": r.get("status"),
                "average_time_ms": timing.get("average"),
                "timing_ms": timing,
                "error_core": r.get("error_core"),
            }
        )
        latency_summary.append({"batch_size": r.get("batch_size"), "average_time_ms": timing.get("average")})
    avg_time_summary.sort(
        key=lambda x: (
            str(x.get("pb_path") or ""),
            int(x.get("batch_size") or 0),
        )
    )

    final_report = {
        "args": vars(args),
        "summary": summary,
        "results": all_reports,
        "average_time_summary": avg_time_summary,
    }

    auto_pb_root.mkdir(parents=True, exist_ok=True)
    report_path = auto_pb_root / "run_report.json"
    report_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] report={report_path}")
    print(f"[OK] summary={summary}")
    print(f"[OK] latency_summary={latency_summary}")

    if failures and args.strict:
        raise RuntimeError("some specs failed, see run_report.json")


if __name__ == "__main__":
    main()
'''
示例用法:
  # 在 MUSA 设备上运行
  python musa_run_pb_graph.py --spec /path/to/*.spec --bs 4096

  '''
