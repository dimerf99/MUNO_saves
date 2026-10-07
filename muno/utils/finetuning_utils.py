import torch
import dill
from copy import deepcopy

MODEL_PARTS = {"core", "liftings", "projections"}

LOAD_ACTIONS = {"load_train", "load_freeze"}
NEW_ACTIONS = {"new_train"}

FINETUNE_ACTIONS = LOAD_ACTIONS | NEW_ACTIONS
INFERENCE_ACTIONS = {"load_freeze"}

FINETUNE_MODE_PRESETS = {
    "train_core_train_adapters": {
        "core": "load_train",
        "liftings": "load_train",
        "projections": "load_train"
    },
    "train_core_train_new_adapters": {
        "core": "load_train",
        "liftings": "new_train",
        "projections": "new_train"
    },

    "freeze_core_train_adapters": {
        "core": "load_freeze",
        "liftings": "load_train",
        "projections": "load_train"
    },
    "freeze_core_train_new_adapters": {
        "core": "load_freeze",
        "liftings": "new_train",
        "projections": "new_train"
    },
    "freeze_core_lift_train_proj": {
        "core": "load_freeze",
        "liftings": "load_freeze",
        "projections": "load_train"
    },
    "freeze_core_lift_train_new_proj": {
        "core": "load_freeze",
        "liftings": "load_freeze",
        "projections": "new_train"
    },
    "freeze_core_proj_train_lift": {
        "core": "load_freeze",
        "liftings": "load_train",
        "projections": "load_freeze"
    },
    "freeze_core_proj_train_new_lift": {
        "core": "load_freeze",
        "liftings": "new_train",
        "projections": "load_freeze"
    },

    "freeze_adapters_train_core": {
        "core": "load_train",
        "liftings": "load_freeze",
        "projections": "load_freeze"
    },
    "freeze_lift_train_core_proj": {
        "core": "load_train",
        "liftings": "load_freeze",
        "projections": "load_train"
    },
    "freeze_proj_train_lift_core": {
        "core": "load_train",
        "liftings": "load_train",
        "projections": "load_freeze"
    }
}
INFERENCE_MODE_PRESETS = {
    "freeze_all": {
        "core": "load_freeze",
        "liftings": "load_freeze",
        "projections": "load_freeze"
    }
}

TRAINABLE_ACTIONS = {"load_train", "new_train"}
FROZEN_ACTIONS = {"load_freeze"}


def get_finetune_preset(mode):
    if mode not in FINETUNE_MODE_PRESETS:
        raise ValueError(
            f"Unknown finetune mode '{mode}'. "
            f"Available modes: {list(FINETUNE_MODE_PRESETS)}"
        )
    return deepcopy(FINETUNE_MODE_PRESETS[mode])


def get_inference_preset(mode):
    if mode not in INFERENCE_MODE_PRESETS:
        raise ValueError(
            f"Unknown inference mode '{mode}'. "
            f"Available modes: {list(INFERENCE_MODE_PRESETS)}"
        )
    return deepcopy(INFERENCE_MODE_PRESETS[mode])


def validate_preset_actions(preset, allowed_actions, config_name):
    missing_parts = MODEL_PARTS - set(preset)
    if missing_parts:
        raise ValueError(f"{config_name} preset is missing model parts: {list(missing_parts)}")

    unknown_parts = set(preset) - MODEL_PARTS
    if unknown_parts:
        raise ValueError(f"{config_name} preset defines unknown model parts: {list(unknown_parts)}")

    for part, action in preset.items():
        if action not in allowed_actions:
            raise ValueError(
                f"{config_name} preset has invalid action '{action}' "
                f"for model part '{part}'. "
                f"Available actions: {list(allowed_actions)}"
            )


def validate_finetune_config(finetune_config, num_tasks):
    if not isinstance(finetune_config, dict):
        raise TypeError("finetune config must be a dictionary")

    if "mode" not in finetune_config:
        raise ValueError("finetune config must define 'mode'")

    if not isinstance(num_tasks, int) or num_tasks < 1:
        raise ValueError(f"num_tasks must be a positive integer, got {num_tasks}")

    if "checkpoint" not in finetune_config:
        raise ValueError("finetune config must define 'checkpoint'")

    mode = finetune_config["mode"]
    if not isinstance(mode, str):
        raise TypeError("finetune.mode must be a string")

    preset = get_finetune_preset(mode)
    validate_preset_actions(
        preset,
        allowed_actions=FINETUNE_ACTIONS,
        config_name="finetune",
    )

    validate_checkpoint_config(
        checkpoint_config=finetune_config["checkpoint"],
        preset=preset,
        num_tasks=num_tasks,
        config_name="finetune",
    )

    return preset


def validate_inference_config(inference_config, num_tasks):
    if not isinstance(inference_config, dict):
        raise TypeError("inference config must be a dictionary")

    if not isinstance(num_tasks, int) or num_tasks < 1:
        raise ValueError(f"num_tasks must be a positive integer, got {num_tasks}")

    if "mode" not in inference_config:
        raise ValueError("inference config must define 'mode'")

    if "checkpoint" not in inference_config:
        raise ValueError("inference config must define 'checkpoint'")

    mode = inference_config["mode"]
    if not isinstance(mode, str):
        raise TypeError("inference.mode must be a string")

    preset = get_inference_preset(mode)
    validate_preset_actions(
        preset,
        allowed_actions=INFERENCE_ACTIONS,
        config_name="inference",
    )

    validate_checkpoint_config(
        checkpoint_config=inference_config["checkpoint"],
        preset=preset,
        num_tasks=num_tasks,
        config_name="inference",
    )

    return preset


def require_checkpoint_part(checkpoint_config, part, num_tasks, config_name):
    if part not in checkpoint_config:
        raise ValueError(f"{config_name}.checkpoint must define '{part}' for load action")

    value = checkpoint_config[part]

    if part == "core":
        if not is_non_empty_string(value):
            raise ValueError(f"{config_name}.checkpoint.core must be a non-empty path string")
        return

    if not isinstance(value, list):
        raise TypeError(f"{config_name}.checkpoint.{part} must be a list of path strings")

    if len(value) != num_tasks:
        raise ValueError(
            f"{config_name}.checkpoint.{part} must contain exactly {num_tasks} "
            f"paths, got {len(value)}"
        )

    invalid_items = [idx for idx, item in enumerate(value) if not is_non_empty_string(item)]

    if invalid_items:
        raise ValueError(
            f"{config_name}.checkpoint.{part} contains invalid path strings "
            f"at indices: {invalid_items}"
        )


def is_non_empty_string(value):
    return isinstance(value, str) and bool(value.strip())


def validate_checkpoint_config(checkpoint_config, preset, num_tasks, config_name):
    if not isinstance(checkpoint_config, dict):
        raise TypeError(f"{config_name}.checkpoint must be a dictionary")

    unknown_keys = set(checkpoint_config) - MODEL_PARTS
    if unknown_keys:
        raise ValueError(f"{config_name}.checkpoint contains unknown keys: {list(unknown_keys)}")

    for part, action in preset.items():
        if action in LOAD_ACTIONS:
            require_checkpoint_part(
                checkpoint_config,
                part=part,
                num_tasks=num_tasks,
                config_name=config_name,
            )
        elif action in NEW_ACTIONS:
            if part in checkpoint_config:
                raise ValueError(
                    f"{config_name}.checkpoint must not define '{part}' "
                    f"because action for this part is '{action}'"
                )
        else:
            raise ValueError(
                f"{config_name} preset has unsupported action '{action}' "
                f"for model part '{part}'"
            )


def load_torch_module(path, map_location="cpu"):
    if not is_non_empty_string(path):
        raise ValueError(f"checkpoint path must be a non-empty string, got {path}")

    module = torch.load(
        f=path,
        map_location=map_location,
        pickle_module=dill,
        weights_only=False
    )

    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Expected checkpoint '{path}' to be a torch.nn.Module, got {type(module)}")

    return module


def load_checkpoint_parts(preset, model_parts, config, map_location="cpu"):
    checkpoint_config = config["checkpoint"]
    liftings, core, projections = model_parts

    if preset["core"] in LOAD_ACTIONS:
        core = load_torch_module(checkpoint_config["core"], map_location=map_location)
    if preset["liftings"] in LOAD_ACTIONS:
        liftings = [load_torch_module(path, map_location=map_location) for path in checkpoint_config["liftings"]]
    if preset["projections"] in LOAD_ACTIONS:
        projections = [load_torch_module(path, map_location=map_location) for path in checkpoint_config["projections"]]

    return liftings, core, projections


def set_requires_grad(module_or_modules, requires_grad):
    if module_or_modules is None:
        return

    if isinstance(module_or_modules, (list, tuple)):
        for module in module_or_modules:
            set_requires_grad(module, requires_grad)
        return

    for param in module_or_modules.parameters():
        param.requires_grad = requires_grad


def action_requires_grad(action):
    if action in TRAINABLE_ACTIONS:
        return True

    if action in FROZEN_ACTIONS:
        return False

    raise ValueError(f"Unsupported trainability action '{action}'")


def apply_trainability_preset(model_parts, preset):
    liftings, core, projections = model_parts

    set_requires_grad(core, action_requires_grad(preset["core"]))
    set_requires_grad(liftings, action_requires_grad(preset["liftings"]))
    set_requires_grad(projections, action_requires_grad(preset["projections"]))

    return liftings, core, projections


def load_finetune_checkpoint(model_parts, finetune_config, num_tasks, map_location="cpu"):
    preset = validate_finetune_config(finetune_config, num_tasks)
    model_parts = load_checkpoint_parts(
        preset,
        model_parts,
        finetune_config,
        map_location=map_location
    )
    model_parts = apply_trainability_preset(model_parts, preset)
    return model_parts


def load_inference_checkpoint(model_parts, inference_config, num_tasks, map_location="cpu"):
    preset = validate_inference_config(inference_config, num_tasks)
    model_parts = load_checkpoint_parts(
        preset,
        model_parts,
        inference_config,
        map_location=map_location
    )
    model_parts = apply_trainability_preset(model_parts, preset)
    return model_parts
