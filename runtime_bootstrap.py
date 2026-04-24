from __future__ import annotations

import os
import warnings


_DEFAULT_TMP_ROOT = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
_DEFAULT_MPLCONFIGDIR = os.path.join(_DEFAULT_TMP_ROOT, "mplconfig_daest")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("TMPDIR", _DEFAULT_TMP_ROOT)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", _DEFAULT_TMP_ROOT)
os.environ.setdefault("MPLCONFIGDIR", _DEFAULT_MPLCONFIGDIR)
os.environ.setdefault("CUDA_DEVICE_MEMORY_SHARED_CACHE", os.path.join(_DEFAULT_TMP_ROOT, "clisa_hami_vgpu.cache"))

os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

warnings.filterwarnings(
    "ignore",
    message=r".*'repr' attribute with value False was provided to the `Field\(\)` function.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*'frozen' attribute with value True was provided to the `Field\(\)` function.*",
)
