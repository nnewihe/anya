"""
Settings for the Pi service, from a TOML file (default /srv/anya/config.toml,
or $ANYA_PI_CONFIG).  Every key is optional; see pi/config.example.toml.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11
    import tomli as tomllib

DEFAULT_PATH = "/srv/anya/config.toml"


@dataclass
class Processing:
    backend: str = "ncnn"            # pose runtime: ncnn / onnx / torch
    device: str = "cpu"
    scale_height: int = 1080         # 0 = native; native 4K x264 on a Pi is hours
    hwaccel: str = "drm"             # ffmpeg -hwaccel for source decodes ("" = off)
    single_decode: bool = True       # both proxies from one source decode
    threads: int = 0                 # torch/ncnn threads; 0 = library default
    keep_inbox_days: int = 14        # copied originals kept this long after a reel


@dataclass
class YouTube:
    enabled: bool = False
    privacy: str = "unlisted"        # unlisted / private / public
    title_prefix: str = "Tennis"
    playlist_id: str = ""
    token: str = ""                  # default <state>/youtube_token.json
    max_attempts: int = 20           # then the job is left as upload_failed


@dataclass
class Config:
    root: Path = Path("/srv/anya")
    site: str = ""                   # default <root>/site
    processing: Processing = field(default_factory=Processing)
    youtube: YouTube = field(default_factory=YouTube)
    poll_s: int = 30

    @property
    def inbox(self):
        return self.root / "inbox"

    @property
    def work(self):
        return self.root / "work"

    @property
    def reels(self):
        return self.root / "reels"

    @property
    def state(self):
        return self.root / "state"

    @property
    def jobs(self):
        return self.state / "jobs"

    @property
    def models(self):
        return self.root / "models"

    @property
    def site_dir(self):
        return Path(self.site) if self.site else self.root / "site"

    @property
    def youtube_token(self):
        return Path(self.youtube.token) if self.youtube.token else self.state / "youtube_token.json"

    def ensure_dirs(self):
        for d in (self.inbox, self.work, self.reels, self.jobs, self.models):
            d.mkdir(parents=True, exist_ok=True)


def _fill(obj, data, where):
    for k, v in (data or {}).items():
        if not hasattr(obj, k):
            raise ValueError(f"unknown setting [{where}] {k}")
        setattr(obj, k, v)


def load(path=None) -> Config:
    path = Path(path or os.environ.get("ANYA_PI_CONFIG", DEFAULT_PATH))
    cfg = Config()
    if not path.is_file():
        return cfg
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    top = {k: v for k, v in data.items() if not isinstance(v, dict)}
    if "root" in top:
        top["root"] = Path(top["root"])
    _fill(cfg, top, "top level")
    _fill(cfg.processing, data.get("processing"), "processing")
    _fill(cfg.youtube, data.get("youtube"), "youtube")
    if cfg.youtube.privacy not in ("unlisted", "private", "public"):
        raise ValueError(f"[youtube] privacy must be unlisted/private/public, "
                         f"not {cfg.youtube.privacy!r}")
    return cfg
