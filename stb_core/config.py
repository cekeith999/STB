import os, json

_DEFAULTS = {
    "profile": "dev",
    "active_provider": "meshy",
    "providers": {},
    "pipeline": {"timeout_s": 900, "poll_interval_s": 3},
    "safety": {"allow_file_ops": False, "whitelist_ops": []},
    "logging": {"level": "INFO", "to_file": True, "path": "logs/stb.log"},
}

def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a

def load_config(repo_root=None):
    # repo_root default = current working dir
    repo_root = repo_root or os.getcwd()
    cfg_path = os.path.join(repo_root, "config", "config.json")
    data = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    cfg = _merge(_DEFAULTS.copy(), data)

    # ensure logs dir exists if logging to file
    log_path = cfg.get("logging", {}).get("path")
    if log_path:
        full = os.path.join(repo_root, log_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        cfg["logging"]["path"] = full

    cfg["_repo_root"] = repo_root  # handy for relative paths
    return cfg
