import os

import cv2
import numpy as np
import pytest
from insightface.app import FaceAnalysis

from app.domain import InvalidImage, MultipleFacesDetected, NoFaceDetected
from app.ml import InsightFaceEmbedder
from scripts.sample_images import face_image_bytes, sample_image_path
from scripts.verify_week3 import run_verification

pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        os.getenv("RUN_REAL_MODEL_TESTS") != "1",
        reason="set RUN_REAL_MODEL_TESTS=1 after downloading the official buffalo_s model",
    ),
]


def test_real_image_embedding_persists_and_matches_through_api(tmp_path) -> None:
    database_url = f"sqlite:///{(tmp_path / 'real-e2e.db').as_posix()}"

    result = run_verification(database_url)

    assert result["model"] == "buffalo_s"
    assert result["embedding_dim"] == 512
    assert result["embedding_dtype"] == "float32"
    assert result["embedding_norm"] == pytest.approx(1.0, abs=1e-5)
    assert result["profile_persisted"] is True
    assert result["check_in_status"] == "success"
    assert result["similarity_score"] == pytest.approx(1.0, abs=1e-4)
    assert result["no_face_http_status"] == 422
    assert result["persisted_statuses"] == ["no_face", "success"]

    assert result["user_history_statuses"] == ["success"]
    assert result["admin_login_http_status"] == 200
    assert result["user_login_http_status"] == 200
    assert result["auth_me_http_status"] == 200
    assert result["invalid_token_http_status"] == 401
    assert result["rbac_forbidden_http_status"] == 403


def test_filtered_runtime_matches_the_full_pack_and_keeps_rejections() -> None:
    full_pack = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    full_pack.prepare(ctx_id=0, det_size=(640, 640))
    embedder = InsightFaceEmbedder(model_name="buffalo_s")

    embedding = embedder.embed(face_image_bytes())

    assert embedding.shape == (512,)
    np.testing.assert_allclose(
        embedding,
        InsightFaceEmbedder(analysis=full_pack).embed(face_image_bytes()),
        rtol=0,
        atol=1e-5,
    )

    group = cv2.imread(str(sample_image_path()))
    _, two_faces = cv2.imencode(".jpg", group[120:440, 240:600])
    _, no_face = cv2.imencode(".png", np.zeros((128, 128, 3), dtype=np.uint8))
    with pytest.raises(MultipleFacesDetected):
        embedder.embed(two_faces.tobytes())
    with pytest.raises(NoFaceDetected):
        embedder.embed(no_face.tobytes())
    with pytest.raises(InvalidImage):
        embedder.embed(b"not-an-image")
