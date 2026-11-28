from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import numpy as np
import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

from app.api.deps import (
    get_embedder,
    get_engine,
    get_session,
    get_session_factory,
)
from app.core.config import get_settings
from app.core.security import create_access_token
from app.domain import FaceEmbedder, InvalidImage, NoFaceDetected, Role, UnmatchedFace
from app.models import Base, CheckInRecordORM, FaceProfileORM, UserORM

JWT_SECRET = "phase2-audit-test-secret-do-not-use-in-production"
FAKE_MODEL_NAME = "phase2-fake-model"


class FakeEmbedder(FaceEmbedder):
    """Embedder giả: không tải InsightFace/ONNX khi chạy test."""

    def __init__(
        self,
        *,
        result: np.ndarray | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error

    @property
    def model_name(self) -> str:
        return FAKE_MODEL_NAME

    def embed(self, image_bytes: bytes) -> np.ndarray:
        # API upload phải chuyển được nội dung file không rỗng tới embedder.
        assert image_bytes

        if self._error is not None:
            raise self._error

        if self._result is None:
            raise AssertionError("FakeEmbedder needs either result or error")

        return np.asarray(self._result, dtype=np.float32)


def _reset_database_dependencies() -> None:
    """Xóa cache engine/session/settings để mỗi test dùng SQLite riêng."""
    if get_engine.cache_info().currsize:
        get_engine().dispose()

    get_session_factory.cache_clear()
    get_engine.cache_clear()
    get_settings.cache_clear()


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    chunk_body = chunk_type + data
    checksum = zlib.crc32(chunk_body) & 0xFFFFFFFF
    return len(data).to_bytes(4, "big") + chunk_body + checksum.to_bytes(4, "big")


def _valid_png_bytes() -> bytes:
    """Tạo PNG RGBA 2x2 hợp lệ chỉ bằng thư viện chuẩn Python."""
    width = 2
    height = 2
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    pixel = b"\xff\x00\x00\xff"  # đỏ, alpha 255
    scanline = b"\x00" + pixel * width  # filter byte + 2 pixels
    image_data = scanline * height

    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(image_data))
        + _png_chunk(b"IEND", b"")
    )


@pytest.fixture
def audit_app(tmp_path, monkeypatch):
    """Tạo FastAPI app + database SQLite tạm cho từng test."""
    database_file = tmp_path / "phase2-audit.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_file.as_posix()}")
    monkeypatch.setenv("JWT_SECRET", JWT_SECRET)
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")

    _reset_database_dependencies()

    # Import sau khi đặt biến môi trường vì app.main tạo app toàn cục khi import.
    from app.main import create_app

    app = create_app()
    Base.metadata.create_all(bind=get_engine())

    yield app

    _reset_database_dependencies()


def _make_embedder_result(vector: list[float] | None = None) -> FakeEmbedder:
    if vector is None:
        vector = [1.0, 0.0]
    return FakeEmbedder(result=np.asarray(vector, dtype=np.float32))


def _create_user(*, email: str = "audit-user@example.com") -> int:
    """Seed user trực tiếp; password không được dùng trong các request test này."""
    with get_session_factory()() as session:
        user = UserORM(
            email=email,
            hashed_password="not-used-by-this-test",
            role="user",
            full_name="Audit Test User",
            created_at=datetime.now(UTC),
        )
        session.add(user)
        session.commit()
        return user.id


def _create_face_profile(user_id: int, embedding: np.ndarray | None = None) -> int:
    """Seed profile có vector trùng với FakeEmbedder để tạo ca SUCCESS."""
    if embedding is None:
        embedding = np.asarray([1.0, 0.0], dtype="<f4")
    else:
        embedding = np.asarray(embedding, dtype="<f4")

    with get_session_factory()() as session:
        profile = FaceProfileORM(
            user_id=user_id,
            embedding=np.ascontiguousarray(embedding).tobytes(),
            embedding_dim=int(embedding.size),
            model_name=FAKE_MODEL_NAME,
            created_at=datetime.now(UTC),
        )
        session.add(profile)
        session.commit()
        return profile.id


def _create_user_token(user_id: int) -> str:
    return create_access_token(
        user_id=user_id,
        role=Role.USER,
        secret=JWT_SECRET,
        algorithm="HS256",
        expires_delta=timedelta(minutes=5),
    )


def _post_checkin(client: TestClient):
    return client.post(
        "/api/v1/checkins",
        files={"image": ("face.png", _valid_png_bytes(), "image/png")},
    )


def _post_invalid_checkin(client: TestClient):
    """Gửi file ảnh rỗng để API phát sinh InvalidImage."""
    return client.post(
        "/api/v1/checkins",
        files={"image": ("empty.png", b"", "image/png")},
    )


def _post_face_registration(client: TestClient, token: str):
    return client.post(
        "/api/v1/face-profiles",
        headers={"Authorization": f"Bearer {token}"},
        files={"image": ("face.png", _valid_png_bytes(), "image/png")},
    )


def _audit_rows() -> list[dict[str, int | str | None]]:
    """Đọc audit bằng Session MỚI, độc lập với Session xử lý HTTP request."""
    with get_session_factory()() as session:
        rows = session.scalars(select(CheckInRecordORM).order_by(CheckInRecordORM.id)).all()
        return [
            {
                "id": row.id,
                "status": row.status,
                "user_id": row.user_id,
                "matched_face_profile_id": row.matched_face_profile_id,
            }
            for row in rows
        ]


def _count_rows(model: type) -> int:
    with get_session_factory()() as session:
        count = session.scalar(select(func.count()).select_from(model))
        return int(count or 0)


def _user_exists(email: str) -> bool:
    with get_session_factory()() as session:
        user_id = session.scalar(select(UserORM.id).where(UserORM.email == email))
        return user_id is not None


def test_no_face_returns_422_and_commits_one_audit_record(audit_app):
    """Ảnh không có mặt: API trả 422 nhưng audit phải được commit."""
    audit_app.dependency_overrides[get_embedder] = lambda: FakeEmbedder(
        error=NoFaceDetected("No face in uploaded image")
    )

    with TestClient(audit_app) as client:
        response = _post_checkin(client)

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "NoFaceDetected"
    assert isinstance(body["record_id"], int)

    rows = _audit_rows()
    assert len(rows) == 1
    assert rows[0]["id"] == body["record_id"]
    assert rows[0]["status"] == "no_face"
    assert rows[0]["user_id"] is None


def test_unmatched_returns_404_and_commits_one_audit_record(audit_app):
    """Không có profile khớp: API trả 404 nhưng audit vẫn phải được commit."""
    audit_app.dependency_overrides[get_embedder] = lambda: _make_embedder_result()
    # Không seed FaceProfileORM: service sẽ đi vào nhánh unmatched.

    with TestClient(audit_app) as client:
        response = _post_checkin(client)

    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "UnmatchedFace"
    assert isinstance(body["record_id"], int)

    rows = _audit_rows()
    assert len(rows) == 1
    assert rows[0]["id"] == body["record_id"]
    assert rows[0]["status"] == "unmatched"
    assert rows[0]["user_id"] is None
    assert rows[0]["matched_face_profile_id"] is None


def test_success_returns_201_and_commits_one_audit_record(audit_app):
    """Vector khớp profile: API trả 201 và lưu một audit success."""
    audit_app.dependency_overrides[get_embedder] = lambda: _make_embedder_result()
    user_id = _create_user()
    profile_id = _create_face_profile(user_id)

    with TestClient(audit_app) as client:
        response = _post_checkin(client)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "success"
    assert body["user_id"] == user_id
    assert body["matched_face_profile_id"] == profile_id
    assert body["full_name"] == "Audit Test User"

    rows = _audit_rows()
    assert len(rows) == 1
    assert rows[0]["id"] == body["id"]
    assert rows[0]["status"] == "success"
    assert rows[0]["user_id"] == user_id
    assert rows[0]["matched_face_profile_id"] == profile_id


def test_register_face_without_face_returns_422_and_creates_no_audit(audit_app):
    """Đăng ký profile không phát hiện mặt: không được tạo check-in audit."""
    audit_app.dependency_overrides[get_embedder] = lambda: FakeEmbedder(
        error=NoFaceDetected("No face in registration image")
    )
    user_id = _create_user(email="register-user@example.com")
    token = _create_user_token(user_id)

    with TestClient(audit_app) as client:
        response = _post_face_registration(client, token)

    assert response.status_code == 422
    assert response.json()["error"] == "NoFaceDetected"
    assert "record_id" not in response.json()
    assert _count_rows(CheckInRecordORM) == 0
    assert _count_rows(FaceProfileORM) == 0


def test_invalid_image_returns_400_without_writes_and_next_checkin_succeeds(
    audit_app,
):
    """Ảnh rỗng trả 400, không tạo audit/dữ liệu mới và request sau vẫn chạy."""
    audit_app.dependency_overrides[get_embedder] = lambda: _make_embedder_result()

    user_id = _create_user(email="invalid-image-followup@example.com")
    _create_face_profile(user_id)

    audit_before = _count_rows(CheckInRecordORM)
    user_before = _count_rows(UserORM)
    profile_before = _count_rows(FaceProfileORM)

    with TestClient(audit_app) as client:
        invalid_response = _post_invalid_checkin(client)

        assert invalid_response.status_code == 400
        assert invalid_response.json()["error"] == "InvalidImage"

        # Request ảnh lỗi không được tạo audit hoặc ghi thêm dữ liệu.
        assert _count_rows(CheckInRecordORM) == audit_before
        assert _count_rows(UserORM) == user_before
        assert _count_rows(FaceProfileORM) == profile_before

        # Kiểm tra Session vẫn hoạt động ở request tiếp theo.
        followup_response = _post_checkin(client)

    assert followup_response.status_code == 201


def test_other_domain_error_after_flush_rolls_back(audit_app):
    """DomainError khác sau flush phải rollback thay vì commit dữ liệu dở."""
    email = "must-rollback-domain@example.com"

    @audit_app.post("/_test/domain-error-after-flush")
    def domain_error_after_flush(
        session: Annotated[Session, Depends(get_session)],
    ) -> None:
        session.add(
            UserORM(
                email=email,
                hashed_password="test-hash",
                role="user",
                full_name="Must Roll Back",
                created_at=datetime.now(UTC),
            )
        )
        session.flush()
        raise InvalidImage("injected domain error after flush")

    with TestClient(audit_app) as client:
        response = client.post("/_test/domain-error-after-flush")

    # InvalidImage được map thành HTTP 400 bởi app/api/errors.py.
    assert response.status_code == 400
    assert response.json()["error"] == "InvalidImage"
    assert not _user_exists(email)
    assert _count_rows(CheckInRecordORM) == 0


@pytest.mark.parametrize(
    ("kind", "expected_status"),
    [("no-face", 422), ("unmatched", 404)],
)
def test_audit_error_without_record_id_rolls_back_flushed_data(
    audit_app,
    kind: str,
    expected_status: int,
):
    """Hai lỗi audit nếu KHÔNG có record_id thì cũng phải rollback."""
    email = f"must-rollback-{kind}@example.com"

    @audit_app.post("/_test/audit-error-without-id/{error_kind}")
    def audit_error_without_id(
        error_kind: str,
        session: Annotated[Session, Depends(get_session)],
    ) -> None:
        session.add(
            UserORM(
                email=email,
                hashed_password="test-hash",
                role="user",
                full_name="Must Roll Back",
                created_at=datetime.now(UTC),
            )
        )
        session.flush()

        if error_kind == "no-face":
            raise NoFaceDetected("missing record id")
        raise UnmatchedFace(0.2)

    with TestClient(audit_app) as client:
        response = client.post(f"/_test/audit-error-without-id/{kind}")

    assert response.status_code == expected_status
    assert not _user_exists(email)
    assert _count_rows(CheckInRecordORM) == 0


def test_unexpected_system_error_after_flush_rolls_back(audit_app):
    """RuntimeError sau flush phải rollback; HTTP mặc định là 500."""
    email = "must-rollback-system@example.com"

    @audit_app.post("/_test/system-error-after-flush")
    def system_error_after_flush(
        session: Annotated[Session, Depends(get_session)],
    ) -> None:
        session.add(
            UserORM(
                email=email,
                hashed_password="test-hash",
                role="user",
                full_name="Must Roll Back",
                created_at=datetime.now(UTC),
            )
        )
        session.flush()
        raise RuntimeError("injected system error after flush")

    with TestClient(audit_app, raise_server_exceptions=False) as client:
        response = client.post("/_test/system-error-after-flush")

    assert response.status_code == 500
    assert not _user_exists(email)
    assert _count_rows(CheckInRecordORM) == 0


def test_commit_failure_rolls_back_and_closes_session(audit_app, monkeypatch):
    """Commit thất bại phải gọi rollback, đóng Session và không lưu dữ liệu."""
    email = "must-rollback-commit-failure@example.com"

    @audit_app.post("/_test/commit-failure")
    def create_then_commit(
        session: Annotated[
            Session,
            Depends(get_session, scope="function"),
        ],
    ) -> dict[str, bool]:

        session.add(
            UserORM(
                email=email,
                hashed_password="test-hash",
                role="user",
                full_name="Commit Failure",
                created_at=datetime.now(UTC),
            )
        )
        # Chứng minh row đã được flush vào transaction trước khi commit lỗi.
        session.flush()
        return {"ok": True}

    counters = {"commit": 0, "rollback": 0, "close": 0}

    def fail_commit(self, *args, **kwargs):
        counters["commit"] += 1
        raise RuntimeError("injected commit failure")

    original_rollback = Session.rollback
    original_close = Session.close

    def track_rollback(self, *args, **kwargs):
        counters["rollback"] += 1
        return original_rollback(self, *args, **kwargs)

    def track_close(self, *args, **kwargs):
        counters["close"] += 1
        return original_close(self, *args, **kwargs)

    monkeypatch.setattr(Session, "commit", fail_commit)
    monkeypatch.setattr(Session, "rollback", track_rollback)
    monkeypatch.setattr(Session, "close", track_close)

    with TestClient(audit_app, raise_server_exceptions=False) as client:
        response = client.post("/_test/commit-failure")

    assert response.status_code == 500
    assert counters["commit"] == 1
    assert counters["rollback"] >= 1
    assert counters["close"] >= 1

    # Query bằng một Session mới sau khi request và cleanup đã kết thúc.
    assert not _user_exists(email)
    assert _count_rows(CheckInRecordORM) == 0


def export_audit_report(report: dict | list) -> None:
    """Chỉ xuất JSON khi người chạy chủ động đặt AUDIT_PROBE_OUT."""
    output_path = os.getenv("AUDIT_PROBE_OUT")

    # Chạy pytest bình thường thì không ghi file.
    if not output_path:
        return

    report_data = dict(report) if isinstance(report, dict) else {"results": report}

    report_data["git_head"] = os.getenv(
        "AUDIT_PROBE_GIT_HEAD",
        report_data.get("git_head", "not-recorded"),
    )
    report_data["fixture_sha256"] = hashlib.sha256(_valid_png_bytes()).hexdigest()

    rendered = json.dumps(
        report_data,
        ensure_ascii=False,
        indent=2,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        rendered + "\n",
        encoding="utf-8",
    )

    print(f"JSON report written to: {destination.resolve()}")


# ===== Append the helpers and benchmark test below export_audit_report() =====


def _count_users_with_prefix(prefix: str) -> int:
    """Count only synthetic users created by one probe case."""
    with get_session_factory()() as session:
        value = session.scalar(
            select(func.count()).select_from(UserORM).where(UserORM.email.like(f"{prefix}%"))
        )
        return int(value or 0)


def _create_probe_user(session: Session, email: str) -> None:
    session.add(
        UserORM(
            email=email,
            hashed_password="phase2-probe-hash",
            role="user",
            full_name="Phase 2 Probe User",
            created_at=datetime.now(UTC),
        )
    )


def _request_with_session_counters(
    monkeypatch,
    send_request,
    *,
    fail_flush: bool = False,
    fail_commit: bool = False,
) -> tuple[object, dict[str, int]]:
    """Run exactly one HTTP request while counting Session lifecycle calls."""
    counters = {"commit": 0, "rollback": 0, "close": 0, "flush": 0}

    original_commit = Session.commit
    original_rollback = Session.rollback
    original_close = Session.close
    original_flush = Session.flush

    with monkeypatch.context() as patch:

        def tracked_commit(self, *args, **kwargs):
            counters["commit"] += 1
            if fail_commit:
                raise RuntimeError("phase2 probe: injected commit failure")
            return original_commit(self, *args, **kwargs)

        def tracked_rollback(self, *args, **kwargs):
            counters["rollback"] += 1
            return original_rollback(self, *args, **kwargs)

        def tracked_close(self, *args, **kwargs):
            counters["close"] += 1
            return original_close(self, *args, **kwargs)

        def tracked_flush(self, *args, **kwargs):
            counters["flush"] += 1
            if fail_flush:
                raise RuntimeError("phase2 probe: injected flush failure")
            return original_flush(self, *args, **kwargs)

        patch.setattr(Session, "commit", tracked_commit)
        patch.setattr(Session, "rollback", tracked_rollback)
        patch.setattr(Session, "close", tracked_close)
        patch.setattr(Session, "flush", tracked_flush)

        response = send_request()

    return response, counters


def _register_dirty_error_route(
    app,
    *,
    path: str,
    email_prefix: str,
    error_kind: str,
) -> None:
    """Register a test-only endpoint that flushes a user and then raises."""
    state = {"index": 0}

    @app.post(path)
    def dirty_error_endpoint(
        session: Annotated[Session, Depends(get_session)],
    ) -> None:
        email = f"{email_prefix}{state['index']}@example.com"
        state["index"] += 1
        _create_probe_user(session, email)
        session.flush()

        if error_kind == "no_face":
            raise NoFaceDetected("phase2 probe: NoFace without record_id")
        if error_kind == "unmatched":
            raise UnmatchedFace(0.2)
        if error_kind == "domain":
            raise InvalidImage("phase2 probe: InvalidImage after flush")
        if error_kind == "system":
            raise RuntimeError("phase2 probe: RuntimeError after flush")
        raise AssertionError(f"Unknown probe error kind: {error_kind}")


def _register_dirty_flush_route(app, *, path: str, email_prefix: str) -> None:
    """Endpoint whose explicit flush is replaced by an injected failure."""
    state = {"index": 0}

    @app.post(path)
    def flush_failure_endpoint(
        session: Annotated[Session, Depends(get_session)],
    ) -> None:
        email = f"{email_prefix}{state['index']}@example.com"
        state["index"] += 1
        _create_probe_user(session, email)
        session.flush()
        return None


def _register_dirty_commit_route(app, *, path: str, email_prefix: str) -> None:
    """Use function-scoped dependency so commit failure occurs before response."""
    state = {"index": 0}

    @app.post(path)
    def commit_failure_endpoint(
        session: Annotated[
            Session,
            Depends(get_session, scope="function"),
        ],
    ) -> dict[str, bool]:
        email = f"{email_prefix}{state['index']}@example.com"
        state["index"] += 1
        _create_probe_user(session, email)
        session.flush()
        return {"ok": True}


def _count_http_statuses(statuses: list[int]) -> dict[str, int]:
    return dict(sorted(Counter(str(status) for status in statuses).items()))


def _run_ten_request_case(
    *,
    name: str,
    client: TestClient,
    monkeypatch,
    send_request,
    expected_status: int,
    dirty_email_prefix: str | None = None,
    fail_flush: bool = False,
    fail_commit: bool = False,
    follow_up_checkin: bool = False,
) -> dict:
    """Run 10 sequential, independent HTTP requests and measure each delta."""
    statuses: list[int] = []
    record_ids: list[int] = []
    request_details: list[dict] = []
    follow_up_statuses: list[int] = []

    audit_delta_total = 0
    dirty_user_delta_total = 0
    rollback_call_count = 0
    session_closed_count = 0
    commit_call_count = 0
    flush_call_count = 0

    engine = get_engine()
    update_count = {"audit": 0}
    table_name = CheckInRecordORM.__tablename__.lower()

    def count_audit_updates(conn, cursor, statement, parameters, context, executemany):
        normalized = statement.lstrip().lower()
        if normalized.startswith("update") and table_name in normalized:
            update_count["audit"] += 1

    event.listen(engine, "before_cursor_execute", count_audit_updates)
    update_count_before = update_count["audit"]

    try:
        for index in range(10):
            audit_before = len(_audit_rows())
            dirty_before = _count_users_with_prefix(dirty_email_prefix) if dirty_email_prefix else 0

            response, lifecycle = _request_with_session_counters(
                monkeypatch,
                send_request,
                fail_flush=fail_flush,
                fail_commit=fail_commit,
            )

            status = int(response.status_code)
            statuses.append(status)
            if status != expected_status:
                raise AssertionError(
                    f"{name} request {index + 1}: expected HTTP "
                    f"{expected_status}, received {status}; body={response.text}"
                )

            try:
                body = response.json()
            except ValueError:
                body = {}

            record_id = body.get("record_id")
            if record_id is None and body.get("status") == "success":
                record_id = body.get("id")
            if isinstance(record_id, int):
                record_ids.append(record_id)

            audit_after = len(_audit_rows())
            audit_delta = audit_after - audit_before
            audit_delta_total += audit_delta

            dirty_after = _count_users_with_prefix(dirty_email_prefix) if dirty_email_prefix else 0
            dirty_delta = dirty_after - dirty_before
            dirty_user_delta_total += dirty_delta

            rollback_call_count += lifecycle["rollback"]
            session_closed_count += lifecycle["close"]
            commit_call_count += lifecycle["commit"]
            flush_call_count += lifecycle["flush"]

            request_details.append(
                {
                    "request": index + 1,
                    "http_status": status,
                    "record_id": record_id,
                    "audit_before": audit_before,
                    "audit_after": audit_after,
                    "audit_delta": audit_delta,
                    "dirty_users_before": dirty_before,
                    "dirty_users_after": dirty_after,
                    "dirty_user_delta": dirty_delta,
                    "rollback_calls": lifecycle["rollback"],
                    "close_calls": lifecycle["close"],
                    "commit_calls": lifecycle["commit"],
                    "flush_calls": lifecycle["flush"],
                }
            )

            if follow_up_checkin:
                # The error endpoint must not poison the next independent request.
                response_follow_up = _post_checkin(client)
                follow_up_statuses.append(int(response_follow_up.status_code))
                if response_follow_up.status_code != 201:
                    raise AssertionError(
                        f"{name} follow-up after request {index + 1}: "
                        f"expected HTTP 201, received "
                        f"{response_follow_up.status_code}; "
                        f"body={response_follow_up.text}"
                    )
    finally:
        event.remove(engine, "before_cursor_execute", count_audit_updates)

    unique_ids = len(set(record_ids))
    return {
        "http_counts": _count_http_statuses(statuses),
        "record_ids": record_ids,
        "unique_record_ids": unique_ids,
        "overwrite_count": len(record_ids) - unique_ids,
        "audit_delta_total": audit_delta_total,
        "dirty_user_delta_total": dirty_user_delta_total,
        "audit_update_statement_count": update_count["audit"] - update_count_before,
        "rollback_call_count": rollback_call_count,
        "session_closed_count": session_closed_count,
        "commit_call_count": commit_call_count,
        "flush_call_count": flush_call_count,
        "followup_http_counts": _count_http_statuses(follow_up_statuses),
        "requests": request_details,
    }


def test_phase2_audit_benchmark_report(audit_app, monkeypatch):
    """Benchmark Base/Fix with 10 independent requests for every scenario."""
    # Prepare one account for registration and one matched profile for success.
    registration_user_id = _create_user(email="phase2-registration-user@example.com")
    registration_token = _create_user_token(registration_user_id)

    success_user_id = _create_user(email="phase2-success-user@example.com")
    _create_face_profile(success_user_id, np.asarray([1.0, 0.0], dtype=np.float32))

    # Test-only endpoints. They are registered only on this in-memory test app.
    route_specs = [
        ("no_face_without_record_id", "/_phase2/no-face-without-id", "no_face"),
        ("unmatched_without_record_id", "/_phase2/unmatched-without-id", "unmatched"),
        ("other_domain_error_after_flush", "/_phase2/domain-error-after-flush", "domain"),
        ("system_error_after_flush", "/_phase2/system-error-after-flush", "system"),
    ]
    dirty_prefixes: dict[str, str] = {}
    route_paths: dict[str, str] = {}

    for case_name, route_path, error_kind in route_specs:
        prefix = f"phase2-probe-{case_name}-"
        dirty_prefixes[case_name] = prefix
        route_paths[case_name] = route_path
        _register_dirty_error_route(
            audit_app,
            path=route_path,
            email_prefix=prefix,
            error_kind=error_kind,
        )

    flush_prefix = "phase2-probe-flush-failure-"
    route_paths["flush_failure"] = "/_phase2/flush-failure"
    dirty_prefixes["flush_failure"] = flush_prefix
    _register_dirty_flush_route(
        audit_app,
        path=route_paths["flush_failure"],
        email_prefix=flush_prefix,
    )

    commit_prefix = "phase2-probe-commit-failure-"
    route_paths["commit_failure"] = "/_phase2/commit-failure"
    dirty_prefixes["commit_failure"] = commit_prefix
    _register_dirty_commit_route(
        audit_app,
        path=route_paths["commit_failure"],
        email_prefix=commit_prefix,
    )

    results: dict[str, dict] = {}

    with TestClient(audit_app, raise_server_exceptions=False) as client:
        # 1. Face registration without a face: 422, no check-in audit/profile.
        audit_app.dependency_overrides[get_embedder] = lambda: FakeEmbedder(
            error=NoFaceDetected("phase2 benchmark: no face during registration")
        )
        profile_before = _count_rows(FaceProfileORM)
        results["face_profile_registration_no_face"] = _run_ten_request_case(
            name="face_profile_registration_no_face",
            client=client,
            monkeypatch=monkeypatch,
            send_request=lambda: _post_face_registration(client, registration_token),
            expected_status=422,
        )
        results["face_profile_registration_no_face"]["face_profile_delta_total"] = (
            _count_rows(FaceProfileORM) - profile_before
        )

        # 2. NoFace with a real audit record_id: each request must add one row.
        audit_app.dependency_overrides[get_embedder] = lambda: FakeEmbedder(
            error=NoFaceDetected("phase2 benchmark: no face")
        )
        results["no_face_with_record_id"] = _run_ten_request_case(
            name="no_face_with_record_id",
            client=client,
            monkeypatch=monkeypatch,
            send_request=lambda: _post_checkin(client),
            expected_status=422,
        )

        # 3. Unmatched with a real audit record_id. Vector is orthogonal to profile.
        audit_app.dependency_overrides[get_embedder] = lambda: _make_embedder_result([0.0, 1.0])
        results["unmatched_with_record_id"] = _run_ten_request_case(
            name="unmatched_with_record_id",
            client=client,
            monkeypatch=monkeypatch,
            send_request=lambda: _post_checkin(client),
            expected_status=404,
        )

        # 4. Successful check-in: each independent request creates one audit.
        audit_app.dependency_overrides[get_embedder] = lambda: _make_embedder_result([1.0, 0.0])
        results["success"] = _run_ten_request_case(
            name="success",
            client=client,
            monkeypatch=monkeypatch,
            send_request=lambda: _post_checkin(client),
            expected_status=201,
        )

        # 5. InvalidImage: ảnh rỗng trả 400, rollback và request tiếp theo vẫn chạy.
        audit_app.dependency_overrides[get_embedder] = lambda: _make_embedder_result([1.0, 0.0])

        results["invalid_image"] = _run_ten_request_case(
            name="invalid_image",
            client=client,
            monkeypatch=monkeypatch,
            send_request=lambda: _post_invalid_checkin(client),
            expected_status=400,
            dirty_email_prefix="phase2-probe-invalid-image-",
            follow_up_checkin=True,
        )

        # 6-11. Dirty-write/error cases. Each is followed by a successful check-in.
        rollback_cases = [
            (
                "no_face_without_record_id",
                422,
                False,
                False,
            ),
            (
                "unmatched_without_record_id",
                404,
                False,
                False,
            ),
            (
                "other_domain_error_after_flush",
                400,
                False,
                False,
            ),
            (
                "system_error_after_flush",
                500,
                False,
                False,
            ),
            (
                "flush_failure",
                500,
                True,
                False,
            ),
            (
                "commit_failure",
                500,
                False,
                True,
            ),
        ]

        for case_name, expected_status, fail_flush, fail_commit in rollback_cases:
            audit_app.dependency_overrides[get_embedder] = lambda: _make_embedder_result([1.0, 0.0])
            results[case_name] = _run_ten_request_case(
                name=case_name,
                client=client,
                monkeypatch=monkeypatch,
                send_request=lambda path=route_paths[case_name]: client.post(path),
                expected_status=expected_status,
                dirty_email_prefix=dirty_prefixes[case_name],
                fail_flush=fail_flush,
                fail_commit=fail_commit,
                follow_up_checkin=True,
            )

    # Tổng hợp đầy đủ kết quả benchmark
    report = {
        "benchmark": "phase2_audit_transaction_probe",
        "requests_per_case": 10,
        "primary_case_count": len(results),
        "primary_request_count": 10 * len(results),
        "followup_request_count": sum(
            sum(item["followup_http_counts"].values()) for item in results.values()
        ),
        "total_http_request_count": 10 * len(results)
        + sum(sum(item["followup_http_counts"].values()) for item in results.values()),
        "database": "SQLite temporary database",
        "git_head": os.getenv(
            "AUDIT_PROBE_GIT_HEAD",
            "not-recorded",
        ),
        "fixture_sha256": hashlib.sha256(_valid_png_bytes()).hexdigest(),
        "results": results,
    }

    # Xuất JSON trước khi kiểm tra assertion
    # Nhờ vậy vẫn giữ được số liệu nếu một assertion thất bại.
    export_audit_report(report)

    # 1. Kiểm tra mỗi trường hợp có đúng 10 request.
    for case_name, result in results.items():
        assert sum(result["http_counts"].values()) == 10, case_name

        assert len(result["requests"]) == 10, case_name

        # Một request có thể gọi Session.close() nhiều lần.
        assert result["session_closed_count"] >= 10, case_name

        # Xác nhận từng request có ít nhất một lần close.
        for request in result["requests"]:
            assert request["close_calls"] >= 1, (
                f"{case_name}, request {request['request']}: Session was not closed"
            )

    # 2. Các trường hợp audit hợp lệ phải tăng đủ 10 bản ghi.
    for case_name in (
        "no_face_with_record_id",
        "unmatched_with_record_id",
        "success",
    ):
        assert results[case_name]["audit_delta_total"] == 10, case_name
        assert results[case_name]["unique_record_ids"] == 10, case_name
        assert results[case_name]["overwrite_count"] == 0, case_name

    # 3. Đăng ký khuôn mặt không có mặt không tạo audit/profile.
    registration = results["face_profile_registration_no_face"]

    assert registration["face_profile_delta_total"] == 0
    assert registration["audit_delta_total"] == 0

    # 4. Kiểm tra nghiêm ngặt trên phiên bản Fix.
    strict_mode = os.getenv("AUDIT_PROBE_STRICT", "0") == "1"

    if strict_mode:
        rollback_cases = (
            "no_face_without_record_id",
            "unmatched_without_record_id",
            "other_domain_error_after_flush",
            "system_error_after_flush",
            "flush_failure",
            "commit_failure",
            "invalid_image",
        )

        for case_name in rollback_cases:
            result = results[case_name]

            assert result["dirty_user_delta_total"] == 0, case_name

            assert result["rollback_call_count"] >= 10, case_name

            assert result["session_closed_count"] >= 10, case_name

    print("\nPHASE 2 AUDIT BENCHMARK COMPLETED")
