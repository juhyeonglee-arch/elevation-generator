"""시험 준비: 사용자 폴더(~/입면도)를 임시 폴더로 바꾸고, 합성 시험 모델을 만든다."""
import os
import pathlib
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

# elevation_ai.jobs 는 불러올 때 ~/입면도 경로를 정하므로, 그 전에 홈 폴더를 바꿔 둔다.
_HOME = tempfile.mkdtemp(prefix="elev_home_")
os.environ["HOME"] = _HOME
os.environ["USERPROFILE"] = _HOME

# 결과 그림을 모아 두는 곳(Actions 산출물로 올림)
ARTIFACTS = pathlib.Path(os.environ.get("ELEV_ARTIFACTS", ROOT / "test-output"))


@pytest.fixture(scope="session")
def artifacts():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS


@pytest.fixture(scope="session")
def sample_model(tmp_path_factory):
    import make_sample_model
    p = tmp_path_factory.mktemp("model") / "sample_site.3dm"
    make_sample_model.main(str(p))
    return p
