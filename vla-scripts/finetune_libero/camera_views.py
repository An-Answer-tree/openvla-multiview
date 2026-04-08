from typing import Sequence


CAMERA_VIEW_MAP = {
    "agentview": "agentview_rgb",
    "topview": "operation_topview_rgb",
    "leftview": "operation_leftview_rgb",
    "rightview": "operation_rightview_rgb",
    "backview": "operation_backview_rgb",
    "leftbackview": "operation_leftbackview_rgb",
    "rightbackview": "operation_rightbackview_rgb",
}
SUPPORTED_CAMERA_NAMES = tuple(CAMERA_VIEW_MAP.keys())
DEFAULT_CAMERA_NAMES = SUPPORTED_CAMERA_NAMES


def resolve_camera_names(camera_names: Sequence[str]) -> tuple[str, ...]:
    """Validates and normalizes a sequence of logical camera names."""
    if isinstance(camera_names, str):
        normalized_camera_names = (camera_names,)
    else:
        normalized_camera_names = tuple(str(camera_name) for camera_name in camera_names)
    if not normalized_camera_names:
        raise ValueError("Expected at least one camera name in `camera_names`.")

    invalid_camera_names = [
        camera_name for camera_name in normalized_camera_names if camera_name not in CAMERA_VIEW_MAP
    ]
    if invalid_camera_names:
        raise ValueError(
            "Unsupported camera_name values "
            f"{invalid_camera_names}. Choose from: {', '.join(SUPPORTED_CAMERA_NAMES)}"
        )

    seen_camera_names = set()
    duplicate_camera_names = []
    for camera_name in normalized_camera_names:
        if camera_name in seen_camera_names and camera_name not in duplicate_camera_names:
            duplicate_camera_names.append(camera_name)
        seen_camera_names.add(camera_name)

    if duplicate_camera_names:
        raise ValueError(f"Duplicate camera_name values are not allowed: {duplicate_camera_names}")

    return normalized_camera_names


def get_camera_views(camera_names: Sequence[str]) -> tuple[str, ...]:
    """Maps logical camera names to HDF5 observation keys."""
    return tuple(CAMERA_VIEW_MAP[camera_name] for camera_name in resolve_camera_names(camera_names))
