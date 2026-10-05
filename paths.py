"""Paths that follow the XDG Base Directory Specification.

One user, one home directory, so the filesystem already separates users and the
server does not need its own identity scheme.

    XDG_CONFIG_HOME   -> config          (default ~/.config/tinymls)
    XDG_DATA_HOME     -> personalization (default ~/.local/share/tinymls)
    XDG_RUNTIME_DIR   -> socket          (no default; unset is reported as an error)

XDG_CONFIG_HOME takes precedence over a config.json in the working directory, so a
user-level file overrides the one shipped with a checkout.
"""

import os

APP_NAME = "tinymls"
FALLBACK_CONFIG_PATH = "config.json"


def _xdg_dir(env_var: str, default_suffix: str) -> str:
    override = os.environ.get(env_var, "").strip()
    base = override if override and os.path.isabs(override) else os.path.expanduser(
        default_suffix
    )
    return os.path.join(base, APP_NAME)


def config_home() -> str:
    return _xdg_dir("XDG_CONFIG_HOME", "~/.config")


def data_home() -> str:
    return _xdg_dir("XDG_DATA_HOME", "~/.local/share")


class RuntimeDirUnavailable(Exception):
    """XDG_RUNTIME_DIR is missing or unusable."""


def runtime_dir() -> str:
    """Directory for the control socket.

    The XDG Base Directory Specification designates XDG_RUNTIME_DIR for sockets
    and other files valid only during a login session. The spec sets no default
    value, so an unset variable is reported rather than silently falling back to
    a location that is not the standard one.
    """
    override = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if not override:
        raise RuntimeDirUnavailable(
            "XDG_RUNTIME_DIR chưa được đặt. Đây là nơi chuẩn để đặt socket "
            "theo XDG Base Directory Specification. Đặt biến này rồi chạy lại, "
            "hoặc truyền --socket / TINYMLS_SOCKET để chỉ định đường dẫn khác."
        )
    if not os.path.isabs(override):
        raise RuntimeDirUnavailable(
            f"XDG_RUNTIME_DIR phải là đường dẫn tuyệt đối, nhận được: {override!r}"
        )
    if not os.path.isdir(override):
        raise RuntimeDirUnavailable(
            f"XDG_RUNTIME_DIR không phải thư mục tồn tại: {override!r}"
        )
    return override


def socket_path() -> str:
    return os.path.join(runtime_dir(), f"{APP_NAME}.sock")


def config_path() -> str:
    """XDG config if it exists, else a config.json in the working directory."""
    candidates = [
        os.path.join(config_home(), FALLBACK_CONFIG_PATH),
        FALLBACK_CONFIG_PATH,
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def personalization_dir() -> str:
    return os.path.join(data_home(), "personalization")
