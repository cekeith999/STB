import os, json

_DEFAULTS = {
    "profile": "dev",
    "active_provider": "meshy",
    "providers": {},
    "pipeline": {"timeout_s": 900, "poll_interval_s": 3},
    "safety": {"allow_file_ops": False, "whitelist_ops": []},
    "logging": {"level": "INFO", "to_file": True, "path": "logs/stb.log"},
    "artifacts_dir": "artifacts",  # default relative folder for generated files
}

def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a

def load_config(repo_root=None):
    repo_root = repo_root or os.getcwd()
    cfg_path = os.path.join(repo_root, "config", "config.json")

    data = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    cfg = _merge(_DEFAULTS.copy(), data)

    # ensure logs dir exists if logging to file, and normalize to absolute path
    log_path_rel = cfg.get("logging", {}).get("path")
    if log_path_rel:
        log_path_abs = os.path.join(repo_root, log_path_rel)
        os.makedirs(os.path.dirname(log_path_abs), exist_ok=True)
        cfg["logging"]["path"] = log_path_abs

    # ensure artifacts dir exists and normalize to absolute path
    art_dir_rel = cfg.get("artifacts_dir")
    if art_dir_rel:
        art_dir_abs = os.path.join(repo_root, art_dir_rel)
        os.makedirs(art_dir_abs, exist_ok=True)
        cfg["artifacts_dir"] = art_dir_abs

    cfg["_repo_root"] = repo_root  # handy for relative paths elsewhere
    return cfg
