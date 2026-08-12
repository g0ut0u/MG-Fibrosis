"""Configuration loading: read yaml configs and merge with default values.

Runtime priority: command-line arguments (argparse) > yaml files > code defaults, overridden layer by layer.
"""
import os
from copy import deepcopy

import yaml

# Project root directory (two levels up from src/ultrasound_mt/config.py to the repository root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")


def _merge(base, override):
    """Deep-merge two dicts: override supersedes base."""
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path):
    """Read a single yaml file and return a dict. Returns {} if the file does not exist."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(data_cfg=None, model_cfg=None, train_cfg=None, paths_cfg=None):
    """Aggregate multiple yaml configs.

    Args:
        data_cfg / model_cfg / train_cfg / paths_cfg: file paths or dicts.
    Convention: each yaml organizes its content under top-level section keys
    (data / model / train / paths), e.g. data.yaml has ``paths:`` and ``data:``
    at the top level. If a yaml's top level is directly the content of that
    section (no nesting), it is also supported (in that case it is handled as
    a corresponding dict passed in).
    """
    cfg = {"paths": {}, "data": {}, "model": {}, "train": {}}

    def _flatten(d, section):
        # d is the content of a yaml/file. If its top level is already that section
        # (no nesting key), use it directly; otherwise extract the top-level section key.
        if section in d:
            return d[section]
        return d

    def _load(item, section):
        if item is None:
            return
        if isinstance(item, dict):
            cfg[section] = _merge(cfg.get(section, {}), _flatten(item, section))
        else:
            d = load_yaml(item)
            for s in ("paths", "data", "model", "train"):
                if s in d:
                    cfg[s] = _merge(cfg.get(s, {}), d[s])

    # Priority: paths_cfg (explicit) > the individual yamls; load the normal yamls first, then paths_cfg overrides at the end
    _load(data_cfg, "data")
    _load(model_cfg, "model")
    _load(train_cfg, "train")
    _load(paths_cfg, "paths")
    return cfg


# Convenience: given a model name, automatically build configs/models/<name>.yaml with configs/data.yaml / train_*.yaml
def resolve_model_yaml(model_name):
    """Return the absolute path to configs/models/<model_name>.yaml (if it exists)."""
    p = os.path.join(CONFIGS_DIR, "models", f"{model_name}.yaml")
    return p if os.path.exists(p) else None
