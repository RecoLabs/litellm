"""Harness coverage for the record/replay transports (LIT-5729).

No proxy and no ``e2e`` marker. A fake in-memory ``Transport`` stands in for
the live one (dependency injection, no monkeypatching): recording must pass
every value through unchanged while writing one redacted interaction file per
call, and replay must serve identical values from the bundle alone - the
fake's call log proves nothing reaches the inner transport - failing hard
(``ReplayMiss``) on any drift in order, verb, or path. The collection-time
gate and report header are pinned here too, including the stale message that
names the bundle's age.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel

from e2e_http import (
    AuthHeaders,
    BinaryStream,
    ProbeResult,
    Result,
    StreamingResponse,
    Success,
)
from fixture_bundle import (
    BUNDLE_FORMAT_VERSION,
    MANIFEST_FILENAME,
    BundleRecorder,
    Interaction,
    LoadedBundle,
    Manifest,
    load_bundle,
    prepare_bundle,
    slug_for_test,
)
from fixture_transport import (
    SESSION_TEST_KEY,
    InvalidFixtureMode,
    RecordingTransport,
    ReplayMiss,
    ReplaySource,
    ReplayTransport,
    current_test_key,
    deterministic_marker,
    enter_non_function_fixture,
    exit_non_function_fixture,
    fixture_mode_collection_error,
    fixture_report_lines,
    parse_fixture_mode,
    replay_leftover_error,
    select_transport,
    wrap_fixture_setup,
)
from transport import Transport

NOW = datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)


class Payload(BaseModel):
    value: str


class Body(BaseModel):
    prompt: str


class Query(BaseModel):
    q: str


STREAMING = StreamingResponse(
    status_code=200,
    body="",
    content_type="text/event-stream",
    chunks=2,
    stream_events=["one", "two"],
    stream_done=True,
)
BINARY = BinaryStream(status_code=200, content_type="audio/mpeg", chunk_count=3, total_bytes=42)
PROBE = ProbeResult(status_code=200, body="alive")


@dataclass
class FakeTransport:
    calls: list[str] = field(default_factory=list)

    def bearer(self, key: str) -> AuthHeaders:
        return AuthHeaders(authorization=f"Bearer {key}")

    @property
    def master(self) -> AuthHeaders:
        return self.bearer("sk-fake-master")

    def _success[R: BaseModel](self, response_type: type[R]) -> Result[R]:
        return Success(status_code=200, data=response_type.model_validate({"value": "live"}))

    def post[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        self.calls.append(f"post {path}")
        return self._success(response_type)

    def get[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        params: BaseModel,
        response_type: type[R],
        timeout: float | None = None,
    ) -> Result[R]:
        self.calls.append(f"get {path}")
        return self._success(response_type)

    def delete[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        json: BaseModel,
        response_type: type[R],
        params: BaseModel | None = None,
    ) -> Result[R]:
        self.calls.append(f"delete {path}")
        return self._success(response_type)

    def patch[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        self.calls.append(f"patch {path}")
        return self._success(response_type)

    def put[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        self.calls.append(f"put {path}")
        return self._success(response_type)

    def stream(self, path: str, *, headers: BaseModel, json: BaseModel) -> StreamingResponse:
        self.calls.append(f"stream {path}")
        return STREAMING

    def stream_binary(
        self, path: str, *, headers: BaseModel, json: BaseModel, chunk_size: int = 8192
    ) -> BinaryStream:
        self.calls.append(f"stream_binary {path}")
        return BINARY

    def send(
        self,
        path: str,
        *,
        headers: BaseModel,
        json: BaseModel,
        params: BaseModel | None = None,
        stream: bool = False,
    ) -> StreamingResponse:
        self.calls.append(f"send {path}")
        return STREAMING

    def probe(self, path: str, *, params: BaseModel) -> ProbeResult:
        self.calls.append(f"probe {path}")
        return PROBE

    def upload[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        form: BaseModel,
        filename: str,
        content: bytes,
        file_content_type: str = "application/jsonl",
        file_field: str = "file",
        params: BaseModel | None = None,
        response_type: type[R],
    ) -> Result[R]:
        self.calls.append(f"upload {path}")
        return self._success(response_type)

    def download(self, path: str, *, headers: BaseModel) -> StreamingResponse:
        self.calls.append(f"download {path}")
        return STREAMING


def make_recorder(root: Path) -> BundleRecorder:
    recorder = prepare_bundle(root)
    assert isinstance(recorder, BundleRecorder)
    return recorder


def replay_source(root: Path) -> ReplaySource:
    loaded = load_bundle(root)
    assert isinstance(loaded, LoadedBundle)
    return ReplaySource(bundle=loaded)


def this_tests_files(root: Path) -> list[Path]:
    slug_dir = root / slug_for_test(current_test_key())
    return sorted(slug_dir.glob("*.json")) if slug_dir.is_dir() else []


def write_manifest(root: Path, recorded_at: datetime) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        format_version=BUNDLE_FORMAT_VERSION, recorded_at=recorded_at, harness_version="abc1234"
    )
    (root / MANIFEST_FILENAME).write_text(manifest.model_dump_json(), encoding="utf-8")


class TestParseFixtureMode:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("live", "live"), ("record", "record"), ("replay", "replay"), ("", "live"), ("  REPLAY  ", "replay")],
    )
    def test_known_values_normalize(self, raw: str, expected: str) -> None:
        assert parse_fixture_mode(raw) == expected

    def test_unknown_value_is_invalid_with_the_original_spelling(self) -> None:
        assert parse_fixture_mode("cached") == InvalidFixtureMode(value="cached")


class TestDeterministicMarker:
    def test_sequence_is_a_pure_function_of_test_and_ordinal(self) -> None:
        """A replay process must regenerate exactly the markers the record
        process generated, so the Nth marker of a test is pinned to a pure
        function of the node id and N."""
        key = current_test_key()
        assert deterministic_marker() == hashlib.sha1(f"{key}#0".encode()).hexdigest()[:12]
        assert deterministic_marker() == hashlib.sha1(f"{key}#1".encode()).hexdigest()[:12]


class TestCurrentTestKey:
    def test_names_this_test_and_strips_the_phase(self) -> None:
        key = current_test_key()
        assert key.endswith("TestCurrentTestKey::test_names_this_test_and_strips_the_phase")
        assert "(call)" not in key

    def test_non_function_fixture_marker_wins_over_a_setup_phase_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression for LIT-5729: pytest sets PYTEST_CURRENT_TEST to the
        triggering test even while a session-scoped fixture's setup body runs.
        The non-function-fixture marker must route those recordings to the
        session bucket, or replay depends on which test happened to be first."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/e2e/batches/test_x.py::test_first (setup)")
        assert current_test_key() == "tests/e2e/batches/test_x.py::test_first"
        enter_non_function_fixture("batch_deployments")
        try:
            assert current_test_key() == SESSION_TEST_KEY
        finally:
            exit_non_function_fixture()
        assert current_test_key() == "tests/e2e/batches/test_x.py::test_first"

    def test_non_function_fixture_marker_wins_during_a_teardown_phase_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Session-scoped fixture teardown runs inside the last test's teardown
        phase, so PYTEST_CURRENT_TEST is set. The marker must still win, or the
        teardown traffic gets stored under whichever test happened to be last."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/e2e/batches/test_x.py::test_last (teardown)")
        enter_non_function_fixture("batch_deployments")
        try:
            assert current_test_key() == SESSION_TEST_KEY
        finally:
            exit_non_function_fixture()

    def test_exit_tolerates_an_empty_stack_after_a_setup_failure(self) -> None:
        """The hook registers the pop finalizer before the fixture body runs, so
        a setup that raises before the paired push finalizer is registered still
        leaves an orphan finalizer. Popping an empty stack must be a no-op or
        scope teardown starts failing tests."""
        exit_non_function_fixture()
        assert current_test_key() == current_test_key()


@dataclass
class RecordedFinalizerRegistrar:
    """Stand-in for a pytest ``FixtureDef``/``SubRequest``: records finalizers
    in registration order and can replay them LIFO exactly like
    ``FixtureDef.finish``, so a hook wrapper's teardown-time push/pop can be
    verified without a live pytest session."""

    finalizers: list[object] = field(default_factory=list)

    def addfinalizer(self, finalizer: object) -> None:
        self.finalizers.append(finalizer)

    def finish(self) -> None:
        while self.finalizers:
            finalizer = self.finalizers.pop()
            assert callable(finalizer)
            finalizer()


@dataclass(frozen=True)
class FixtureStub:
    scope: str
    argname: str


class TestWrapFixtureSetup:
    """The hook wrapper the shared conftest installs (LIT-5729): mechanism
    coverage that pins how setup and teardown of a non-function-scoped fixture
    are wrapped, using stubs that expose ``FixtureDef``/``SubRequest`` behavior
    without a live pytest session."""

    def test_function_scoped_fixture_never_touches_the_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/e2e/suite.py::test_fn (setup)")
        registrar = RecordedFinalizerRegistrar()
        generator = wrap_fixture_setup(FixtureStub(scope="function", argname="scoped_key"), registrar)
        next(generator)
        assert current_test_key() == "tests/e2e/suite.py::test_fn"
        with pytest.raises(StopIteration):
            generator.send(None)
        assert registrar.finalizers == []

    @pytest.mark.parametrize("scope", ["session", "module", "class", "package"])
    def test_non_function_scope_marks_setup_and_wires_teardown_around_the_body(
        self, scope: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setup phase (during the yield): marker is active, and the pop
        finalizer is already registered before the fixture body runs, so it
        will run LAST at teardown. Teardown phase (finalizers replay LIFO): the
        marker is active during the fixture body's own yield-teardown, then
        popped last, so the stack is empty after teardown even when
        PYTEST_CURRENT_TEST is still set to the triggering test."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/e2e/suite.py::test_first (setup)")
        registrar = RecordedFinalizerRegistrar()
        generator = wrap_fixture_setup(FixtureStub(scope=scope, argname="session_fx"), registrar)
        next(generator)
        assert current_test_key() == SESSION_TEST_KEY

        body_teardown_key: dict[str, str] = {}
        registrar.addfinalizer(lambda: body_teardown_key.update(observed=current_test_key()))

        with pytest.raises(StopIteration):
            generator.send(None)
        assert current_test_key() == "tests/e2e/suite.py::test_first"

        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/e2e/suite.py::test_last (teardown)")
        registrar.finish()
        assert body_teardown_key["observed"] == SESSION_TEST_KEY
        assert current_test_key() == "tests/e2e/suite.py::test_last"

    def test_setup_failure_still_leaves_the_stack_clean_after_scope_teardown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the fixture body raises before yielding, the paired push
        finalizer is never registered but the pop finalizer is: replaying
        finalizers must still leave the marker stack empty."""
        monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/e2e/suite.py::test_x (setup)")
        registrar = RecordedFinalizerRegistrar()
        generator = wrap_fixture_setup(FixtureStub(scope="session", argname="broken_fx"), registrar)
        next(generator)
        assert current_test_key() == SESSION_TEST_KEY
        with pytest.raises(RuntimeError, match="setup exploded"):
            generator.throw(RuntimeError("setup exploded"))
        assert current_test_key() == "tests/e2e/suite.py::test_x"
        registrar.finish()
        assert current_test_key() == "tests/e2e/suite.py::test_x"


class TestRecordingTransport:
    def test_passes_the_result_through_and_writes_one_file_per_call(self, tmp_path: Path) -> None:
        fake = FakeTransport()
        root = tmp_path / "bundle"
        recording: Transport = RecordingTransport(inner=fake, recorder=make_recorder(root))
        result = recording.post(
            "/model/new", headers=fake.master, json=Body(prompt="x"), response_type=Payload
        )
        assert result == Success(status_code=200, data=Payload(value="live"))
        assert fake.calls == ["post /model/new"]
        files = this_tests_files(root)
        assert [file.name for file in files] == ["0000-post-model-new.json"]
        interaction = Interaction.model_validate_json(files[0].read_text(encoding="utf-8"))
        assert interaction.request.method == "post"
        assert interaction.request.path == "/model/new"

    def test_redacts_auth_header_values_in_the_recorded_request(self, tmp_path: Path) -> None:
        fake = FakeTransport()
        root = tmp_path / "bundle"
        recording: Transport = RecordingTransport(inner=fake, recorder=make_recorder(root))
        headers = AuthHeaders.model_validate(
            {"authorization": "Bearer sk-secret", "x-litellm-api-key": "sk-other"}
        )
        recording.post("/key/generate", headers=headers, json=Body(prompt="x"), response_type=Payload)
        interaction = Interaction.model_validate_json(
            this_tests_files(root)[0].read_text(encoding="utf-8")
        )
        assert interaction.request.headers == {
            "authorization": "<redacted>",
            "x-litellm-api-key": "<redacted>",
        }
        assert "sk-secret" not in this_tests_files(root)[0].read_text(encoding="utf-8")

    def test_upload_records_a_content_digest_not_the_bytes(self, tmp_path: Path) -> None:
        fake = FakeTransport()
        root = tmp_path / "bundle"
        recording: Transport = RecordingTransport(inner=fake, recorder=make_recorder(root))
        recording.upload(
            "/v1/files",
            headers=fake.master,
            form=Query(q="batch"),
            filename="batch.jsonl",
            content=b'{"custom_id": "1"}',
            response_type=Payload,
        )
        interaction = Interaction.model_validate_json(
            this_tests_files(root)[0].read_text(encoding="utf-8")
        )
        assert interaction.request.file_name == "batch.jsonl"
        assert interaction.request.file_bytes == len(b'{"custom_id": "1"}')
        assert interaction.request.file_sha256 is not None
        assert "custom_id" not in interaction.request.model_dump_json()


class TestReplayTransport:
    def test_serves_recorded_values_without_touching_the_inner_transport(
        self, tmp_path: Path
    ) -> None:
        fake = FakeTransport()
        root = tmp_path / "bundle"
        recording: Transport = RecordingTransport(inner=fake, recorder=make_recorder(root))
        recorded_post = recording.post(
            "/model/new", headers=fake.master, json=Body(prompt="x"), response_type=Payload
        )
        recorded_get = recording.get(
            "/v1/models", headers=fake.master, params=Query(q="all"), response_type=Payload
        )
        recorded_stream = recording.stream(
            "/chat/completions", headers=fake.master, json=Body(prompt="hi")
        )
        recorded_probe = recording.probe("/health/liveliness", params=Query(q="1"))
        recorded_binary = recording.stream_binary(
            "/v1/audio/speech", headers=fake.master, json=Body(prompt="say")
        )
        calls_after_record = list(fake.calls)

        replay: Transport = ReplayTransport(source=replay_source(root), master_key="sk-1234")
        assert (
            replay.post("/model/new", headers=replay.master, json=Body(prompt="x"), response_type=Payload)
            == recorded_post
        )
        assert (
            replay.get("/v1/models", headers=replay.master, params=Query(q="all"), response_type=Payload)
            == recorded_get
        )
        assert (
            replay.stream("/chat/completions", headers=replay.master, json=Body(prompt="hi"))
            == recorded_stream
        )
        assert replay.probe("/health/liveliness", params=Query(q="1")) == recorded_probe
        assert (
            replay.stream_binary("/v1/audio/speech", headers=replay.master, json=Body(prompt="say"))
            == recorded_binary
        )
        assert fake.calls == calls_after_record

    def test_mismatched_call_names_recorded_and_actual(self, tmp_path: Path) -> None:
        fake = FakeTransport()
        root = tmp_path / "bundle"
        recording: Transport = RecordingTransport(inner=fake, recorder=make_recorder(root))
        recording.post("/model/new", headers=fake.master, json=Body(prompt="x"), response_type=Payload)
        replay: Transport = ReplayTransport(source=replay_source(root), master_key="sk-1234")
        with pytest.raises(ReplayMiss, match=r"recorded post /model/new, test made get /v1/models"):
            replay.get("/v1/models", headers=replay.master, params=Query(q="all"), response_type=Payload)

    def test_exhausted_recording_names_the_call_count(self, tmp_path: Path) -> None:
        fake = FakeTransport()
        root = tmp_path / "bundle"
        recording: Transport = RecordingTransport(inner=fake, recorder=make_recorder(root))
        recording.post("/model/new", headers=fake.master, json=Body(prompt="x"), response_type=Payload)
        replay: Transport = ReplayTransport(source=replay_source(root), master_key="sk-1234")
        replay.post("/model/new", headers=replay.master, json=Body(prompt="x"), response_type=Payload)
        with pytest.raises(ReplayMiss, match=r"call #2 \(post /model/new\) has no recorded interaction \(1 recorded"):
            replay.post("/model/new", headers=replay.master, json=Body(prompt="x"), response_type=Payload)


class TestReplayLeftover:
    def test_fully_consumed_recording_leaves_nothing(self, tmp_path: Path) -> None:
        fake = FakeTransport()
        root = tmp_path / "bundle"
        recording: Transport = RecordingTransport(inner=fake, recorder=make_recorder(root))
        recording.post("/model/new", headers=fake.master, json=Body(prompt="x"), response_type=Payload)
        source = replay_source(root)
        replay: Transport = ReplayTransport(source=source, master_key="sk-1234")
        replay.post("/model/new", headers=replay.master, json=Body(prompt="x"), response_type=Payload)
        assert source.leftover_error(current_test_key()) is None

    def test_unconsumed_trailing_interactions_name_the_next_call(self, tmp_path: Path) -> None:
        fake = FakeTransport()
        root = tmp_path / "bundle"
        recording: Transport = RecordingTransport(inner=fake, recorder=make_recorder(root))
        recording.post("/model/new", headers=fake.master, json=Body(prompt="x"), response_type=Payload)
        recording.probe("/health/liveliness", params=Query(q="1"))
        source = replay_source(root)
        replay: Transport = ReplayTransport(source=source, master_key="sk-1234")
        replay.post("/model/new", headers=replay.master, json=Body(prompt="x"), response_type=Payload)
        error = source.leftover_error(current_test_key())
        assert error is not None
        assert "1 of 2 recorded interactions never consumed" in error
        assert "next is probe /health/liveliness" in error
        assert "re-record with E2E_FIXTURE_MODE=record" in error

    def test_test_without_recordings_has_no_leftover(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        make_recorder(root)
        assert replay_source(root).leftover_error("suite.py::test_never_recorded") is None

    def test_inert_outside_replay_mode(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"
        assert replay_leftover_error(mode_raw="", bundle_dir=missing, test_key="k") is None
        assert replay_leftover_error(mode_raw="record", bundle_dir=missing, test_key="k") is None

    def test_replay_mode_reads_the_shared_bundle(self, tmp_path: Path) -> None:
        fake = FakeTransport()
        root = tmp_path / "bundle"
        recording: Transport = RecordingTransport(inner=fake, recorder=make_recorder(root))
        recording.post("/model/new", headers=fake.master, json=Body(prompt="x"), response_type=Payload)
        error = replay_leftover_error(mode_raw="replay", bundle_dir=root, test_key=current_test_key())
        assert error is not None
        assert "1 of 1 recorded interactions never consumed" in error


class TestSelectTransport:
    def test_live_returns_the_live_transport_untouched(self, tmp_path: Path) -> None:
        fake = FakeTransport()
        for mode_raw in ("live", ""):
            assert (
                select_transport(fake, mode_raw=mode_raw, bundle_dir=tmp_path / "b", master_key="sk")
                is fake
            )

    def test_record_wraps_live_and_starts_a_fresh_bundle(self, tmp_path: Path) -> None:
        fake = FakeTransport()
        root = tmp_path / "bundle"
        write_manifest(root, NOW - timedelta(days=30))
        (root / "old-test-slug").mkdir()
        (root / "old-test-slug" / "0000-post-old.json").write_text("{}", encoding="utf-8")
        selected = select_transport(fake, mode_raw="record", bundle_dir=root, master_key="sk")
        assert isinstance(selected, RecordingTransport)
        assert selected.inner is fake
        assert {entry.name for entry in root.iterdir()} == {MANIFEST_FILENAME}

    def test_replay_builds_a_transport_from_the_bundle_alone(self, tmp_path: Path) -> None:
        fake = FakeTransport()
        root = tmp_path / "bundle"
        make_recorder(root)
        selected = select_transport(fake, mode_raw="replay", bundle_dir=root, master_key="sk-master")
        assert isinstance(selected, ReplayTransport)
        assert selected.master == AuthHeaders(authorization="Bearer sk-master")

    def test_invalid_mode_raises_naming_the_value(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="cached"):
            select_transport(
                FakeTransport(), mode_raw="cached", bundle_dir=tmp_path / "b", master_key="sk"
            )


class TestCollectionGate:
    def test_invalid_mode_names_the_value_and_the_choices(self, tmp_path: Path) -> None:
        assert (
            fixture_mode_collection_error("cached", tmp_path, now=NOW)
            == "E2E_FIXTURE_MODE='cached' is not one of live, record, replay"
        )

    @pytest.mark.parametrize("mode_raw", ["live", "", "record"])
    def test_live_and_record_never_block_collection(self, mode_raw: str, tmp_path: Path) -> None:
        assert fixture_mode_collection_error(mode_raw, tmp_path / "missing", now=NOW) is None

    def test_replay_with_no_bundle_says_how_to_record_one(self, tmp_path: Path) -> None:
        reason = fixture_mode_collection_error("replay", tmp_path / "missing", now=NOW)
        assert reason is not None
        assert f"no {MANIFEST_FILENAME}" in reason
        assert "E2E_FIXTURE_MODE=record" in reason

    def test_stale_replay_bundle_fails_naming_its_age(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        write_manifest(root, NOW - timedelta(days=9, hours=5))
        reason = fixture_mode_collection_error("replay", root, now=NOW)
        assert reason is not None
        assert "age 9d5h exceeds the 7-day limit" in reason
        assert "re-record with E2E_FIXTURE_MODE=record" in reason

    def test_fresh_replay_bundle_collects(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        write_manifest(root, NOW - timedelta(days=2))
        assert fixture_mode_collection_error("replay", root, now=NOW) is None


class TestReportHeader:
    def test_live_mode_prints_nothing(self, tmp_path: Path) -> None:
        assert fixture_report_lines("live", tmp_path, now=NOW) == []
        assert fixture_report_lines("", tmp_path, now=NOW) == []

    def test_record_and_replay_name_the_bundle(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        recorded_at = NOW - timedelta(days=1)
        write_manifest(root, recorded_at)
        assert fixture_report_lines("record", root, now=NOW) == [
            f"e2e fixture mode: record -> {root}"
        ]
        replay_lines = fixture_report_lines("replay", root, now=NOW)
        assert len(replay_lines) == 1
        assert "replay" in replay_lines[0]
        assert recorded_at.isoformat() in replay_lines[0]
