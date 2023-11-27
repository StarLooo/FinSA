def update_configs(configs, **kwargs):
    if not isinstance(configs, list) and not isinstance(configs, tuple):
        configs = [configs]
    for config in configs:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
