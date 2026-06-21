"""
ai_detect_common.py  v1.2  –  shared library for the AI-image-detection dataset build.

Upload by NB00, downloaded identically by every other notebook. The canonical
preprocess and schema must never branch on the label – real and AI go through
the same code, or the dataset leaks.

v1.2 changes: make_seed now accepts master_seed so one variable in NB00 controls
the entire random state of the build.
"""
from __future__ import annotations
import io, os, json, time, uuid, getpass, hashlib, tempfile, datetime
from collections import Counter
from PIL import Image, ImageOps
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from huggingface_hub import (HfApi, hf_hub_download, list_repo_files, CommitOperationAdd)
    from huggingface_hub.utils import HfHubHTTPError
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False
    class HfHubHTTPError(Exception): pass

PIPELINE_VERSION = "1.2"
TARGET = 512
JPEG_QUALITY = 95

# ─────────────────────────── CANONICAL PREPROCESS ────────────────────────────
def canonical_preprocess(img: "Image.Image") -> bytes:
    """Single transform applied to EVERY image, real or AI. Never branch on label."""
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((TARGET, TARGET), Image.LANCZOS)
    img = Image.frombytes("RGB", img.size, img.tobytes())
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
    buf.seek(0); img = Image.open(buf).convert("RGB"); img.load()
    out = io.BytesIO()
    img.save(out, format="PNG", compress_level=6)
    return out.getvalue()

def decode_png(b: bytes) -> "Image.Image":
    im = Image.open(io.BytesIO(b)); im.load(); return im

def png_is_canonical(png_bytes: bytes):
    try:
        im = Image.open(io.BytesIO(png_bytes)); im.load()
    except Exception as e:
        return False, f"undecodable: {e}"
    if im.format != "PNG":      return False, f"format={im.format}"
    if im.mode != "RGB":        return False, f"mode={im.mode}"
    if im.size != (TARGET, TARGET): return False, f"size={im.size}"
    if im.info.get("icc_profile"): return False, "has ICC profile"
    return True, "ok"

# ─────────────────────────── SEEDS / HASHES / TIME ───────────────────────────
def make_seed(model: str, source_real_id: str, master_seed: int = 0) -> int:
    """Deterministic seed for (model, image, master_seed).
    Change master_seed in NB00 to produce a completely different set of images."""
    h = hashlib.sha256(f"{master_seed}:{model}:{source_real_id}".encode()).hexdigest()
    return int(h[:15], 16) % (2**31 - 1)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def get_device_dtype():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.float16
        return "cpu", torch.float32
    except Exception:
        return "cpu", None

# ─────────────────────────── FROZEN SCHEMAS ──────────────────────────────────
SCHEMA = pa.schema([
    ("image_id",         pa.string()),
    ("source_real_id",   pa.string()),
    ("label",            pa.int8()),
    ("generator",        pa.string()),
    ("source_dataset",   pa.string()),
    ("prompt",           pa.string()),
    ("image",            pa.binary()),
    ("width",            pa.int16()),
    ("height",           pa.int16()),
    ("orig_width",       pa.int32()),
    ("orig_height",      pa.int32()),
    ("gen_model_id",     pa.string()),
    ("gen_steps",        pa.int16()),
    ("gen_guidance",     pa.float32()),
    ("gen_native_res",   pa.int16()),
    ("seed",             pa.int64()),
    ("caption_model",    pa.string()),
    ("pipeline_version", pa.string()),
    ("sha256",           pa.string()),
    ("created_utc",      pa.string()),
])

CAPTION_SCHEMA = pa.schema([
    ("source_real_id", pa.string()),
    ("caption",        pa.string()),
    ("raw_caption",    pa.string()),
    ("caption_model",  pa.string()),
    ("caption_task",   pa.string()),
    ("n_tokens",       pa.int16()),
    ("created_utc",    pa.string()),
])

def empty_row() -> dict:
    return {f.name: None for f in SCHEMA}

# ─────────────────────────── TOKEN / CONFIG / HF I/O ─────────────────────────
def load_hf_token() -> str:
    try:
        from kaggle_secrets import UserSecretsClient
        t = UserSecretsClient().get_secret("HF_TOKEN")
        if t: return t.strip()
    except Exception: pass
    for k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k): return os.environ[k].strip()
    return getpass.getpass("HF write token: ").strip()

def robust_commit(api, repo_id, operations, msg, retries=6):
    delay = 5.0
    for attempt in range(1, retries + 1):
        try:
            api.create_commit(repo_id=repo_id, repo_type="dataset",
                              operations=operations, commit_message=msg)
            return True
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if attempt == retries or code not in (429, 500, 502, 503, 504, None): raise
            print(f"  commit retry {attempt}/{retries} (HTTP {code}); sleep {delay:.0f}s")
            time.sleep(delay); delay = min(delay * 2, 300)
        except Exception as e:
            if attempt == retries: raise
            print(f"  commit retry {attempt}/{retries} ({type(e).__name__}); sleep {delay:.0f}s")
            time.sleep(delay); delay = min(delay * 2, 300)

def read_config(repo_id, token) -> dict:
    p = hf_hub_download(repo_id, "config.json", repo_type="dataset", token=token)
    with open(p) as f: return json.load(f)

def list_shards(repo_id, prefix, token):
    files = list_repo_files(repo_id, repo_type="dataset", token=token)
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))

def reconcile_ids(repo_id, prefix, token) -> set:
    done = set()
    for f in list_shards(repo_id, prefix, token):
        try:
            local = hf_hub_download(repo_id, f, repo_type="dataset", token=token)
            tbl = pq.read_table(local, columns=["source_real_id"])
            done.update(tbl.column("source_real_id").to_pylist())
        except Exception as e:
            print(f"  WARN could not read {f}: {e}")
    return done

def count_rows(repo_id, prefix, token) -> int:
    n = 0
    for f in list_shards(repo_id, prefix, token):
        local = hf_hub_download(repo_id, f, repo_type="dataset", token=token)
        n += pq.ParquetFile(local).metadata.num_rows
    return n

# ─────────────────────────── SHARD WRITER ────────────────────────────────────
class ShardWriter:
    def __init__(self, api, repo_id, prefix, schema=None,
                 commit_interval=1200, max_rows=500, token=None):
        self.api = api; self.repo_id = repo_id
        self.prefix = prefix.rstrip("/") + "/"
        self.tag = self.prefix.strip("/").split("/")[-1]
        self.schema = schema if schema is not None else SCHEMA
        self.commit_interval = commit_interval; self.max_rows = max_rows
        self.token = token; self.session = uuid.uuid4().hex[:8]
        self.seq = 0; self.buf = []; self.last_flush = time.time()
        self.total_committed = 0

    def add(self, row: dict):
        self.buf.append(row)
        if len(self.buf) >= self.max_rows or (time.time() - self.last_flush) >= self.commit_interval:
            self.flush()

    def maybe_flush(self):
        if self.buf and (time.time() - self.last_flush) >= self.commit_interval:
            self.flush()

    def flush(self):
        if not self.buf: return
        tbl = pa.Table.from_pylist(self.buf, schema=self.schema)
        fname = f"{self.prefix}{self.tag}-{self.session}-{self.seq:05d}.parquet"
        tmp = os.path.join(tempfile.gettempdir(), os.path.basename(fname))
        pq.write_table(tbl, tmp, compression="zstd")
        op = CommitOperationAdd(path_in_repo=fname, path_or_fileobj=tmp)
        robust_commit(self.api, self.repo_id, [op], f"add {len(self.buf)} rows -> {fname}")
        self.total_committed += len(self.buf)
        print(f"  committed {len(self.buf)} rows ({self.total_committed} total) -> {fname}")
        self.seq += 1; self.buf = []; self.last_flush = time.time()
        try: os.remove(tmp)
        except OSError: pass

    def close(self): self.flush()

# ─────────────────────────── VERIFIER (NB01) ─────────────────────────────────
def verify_real_stage(repo_id, token, config, sample=200):
    import random as _r
    print("=" * 64)
    print("NB01 VERIFIER  –  real-image stage")
    print("=" * 64)
    problems, warns = [], []

    try: cfg = read_config(repo_id, token)
    except Exception as e:
        print("FAIL: cannot read config.json:", e); return False
    if cfg.get("pipeline_version") != PIPELINE_VERSION:
        problems.append(f"pipeline_version mismatch: config={cfg.get('pipeline_version')} module={PIPELINE_VERSION}")

    shards = list_shards(repo_id, "real/", token)
    print(f"real/ shards found: {len(shards)}")
    if not shards:
        print("FAIL: no real/ shards"); return False

    ids, srcs, shas, n_rows = [], [], [], 0
    for f in shards:
        local = hf_hub_download(repo_id, f, repo_type="dataset", token=token)
        t = pq.read_table(local, columns=["image_id","source_dataset","sha256","label",
                                          "generator","source_real_id","width","height","pipeline_version"])
        n_rows += t.num_rows
        ids   += t.column("image_id").to_pylist()
        srcs  += t.column("source_dataset").to_pylist()
        shas  += t.column("sha256").to_pylist()
        if set(t.column("label").to_pylist()) - {0}:
            problems.append(f"{f}: non-zero label in real stage")
        if set(t.column("generator").to_pylist()) - {"real"}:
            problems.append(f"{f}: generator != 'real'")
        if set(t.column("width").to_pylist()) - {TARGET} or set(t.column("height").to_pylist()) - {TARGET}:
            problems.append(f"{f}: width/height not all {TARGET}")
        if set(t.column("pipeline_version").to_pylist()) - {PIPELINE_VERSION}:
            problems.append(f"{f}: pipeline_version mismatch")
        bad = sum(1 for a, b in zip(t.column("image_id").to_pylist(), t.column("source_real_id").to_pylist()) if a != b)
        if bad: problems.append(f"{f}: {bad} rows where source_real_id != image_id")

    print(f"total real rows: {n_rows}")
    targets = config["real_sources"]
    exp_total = sum(targets.values())
    if n_rows != exp_total:
        warns.append(f"row count {n_rows} != target {exp_total} (fine if still running)")
    cc = Counter(srcs)
    for s, want in targets.items():
        got = cc.get(s, 0); print(f"  {s}: {got} / {want}")
        if got < want: warns.append(f"{s} under target: {got}/{want}")

    dup_ids = [k for k, v in Counter(ids).items() if v > 1]
    if dup_ids: problems.append(f"{len(dup_ids)} duplicate image_id(s)")
    dup_sha = [k for k, v in Counter(shas).items() if v > 1]
    if dup_sha: warns.append(f"{len(dup_sha)} duplicate image bytes (sha256) – NB09 will dedup")

    bad_canon, bad_sha, checked = 0, 0, 0
    sample_shards = shards if len(shards) <= 3 else _r.sample(shards, 3)
    per = max(1, sample // len(sample_shards))
    for f in sample_shards:
        local = hf_hub_download(repo_id, f, repo_type="dataset", token=token)
        t = pq.read_table(local, columns=["image","sha256"]); m = t.num_rows
        idxs = range(m) if m <= per else _r.sample(range(m), per)
        imgs, sh = t.column("image"), t.column("sha256")
        for i in idxs:
            b = imgs[i].as_py(); checked += 1
            ok, _ = png_is_canonical(b)
            if not ok: bad_canon += 1
            if sha256_bytes(b) != sh[i].as_py(): bad_sha += 1
    print(f"sampled {checked} images: canonical_fail={bad_canon}, sha_mismatch={bad_sha}")
    if bad_canon: problems.append(f"{bad_canon}/{checked} images not canonical")
    if bad_sha:   problems.append(f"{bad_sha}/{checked} images sha256 mismatch")

    print("-" * 64)
    for w in warns: print("WARN:", w)
    if problems:
        print("\nRESULT: FAIL")
        for p in problems: print("  –", p)
        return False
    print("\nRESULT: PASS  –  real stage looks correct.")
    return True
