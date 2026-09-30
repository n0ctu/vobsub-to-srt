import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vobsub_to_srt import web  # noqa: E402
from vobsub_to_srt.job import JobResult  # noqa: E402

IDX = b"# VobSub index\nsize: 720x576\nid: en, index: 0\n" + b"".join(
    f"timestamp: 00:00:{i:02d}:000, filepos: {i:09x}\n".encode() for i in range(5))


def fake_run_job_sync(idx_path, config, progress):
    progress({"event": "probe", "db": "calm-sable-0000", "mode": "exact", "coverage": 1.0})
    progress({"event": "cues", "items": [{"i": 0, "start": 0, "end": 1000, "text": "Hello", "src": "nocr"}], "resolved": 1})
    progress({"event": "done", "unresolved": 0, "seconds": 0.1})
    return JobResult(srt="1\n00:00:00,000 --> 00:00:01,000\nHello\n", report={"by_source": {"nocr": 5}},
                     unresolved=0, vlm_used=3, budget_exhausted=False)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "DATA", tmp_path)
    monkeypatch.setattr(web.jobmod, "run_job_sync", fake_run_job_sync)
    web.jobs.clear()
    web.limiter.__init__()
    with TestClient(web.app) as c:
        yield c


def submit(client, idx=IDX, idx_name="movie.idx", sub_name="movie.sub"):
    return client.post("/api/jobs", files={"idx": (idx_name, idx), "sub": (sub_name, b"\x00" * 100)})


def wait_done(client, job_id):
    for _ in range(100):
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_upload_convert_download(client):
    r = submit(client)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    j = wait_done(client, job_id)
    assert j["status"] == "done" and j["unresolved"] == 0 and j["vlm_used"] == 3
    srt = client.get(f"/api/jobs/{job_id}/srt")
    assert srt.status_code == 200 and "Hello" in srt.text
    assert srt.headers["content-disposition"].endswith('filename="movie.srt"')
    events = [json.loads(l[6:]) for l in client.get(f"/api/jobs/{job_id}/events").text.splitlines() if l.startswith("data: ")]
    assert [e["event"] for e in events] == ["queued", "started", "probe", "cues", "done"]
    assert events[3]["items"][0]["text"] == "Hello"
    assert all("db" not in e and "srt" not in e for e in events)          # nothing internal leaks
    assert not (web.DATA / "jobs" / job_id / "movie.sub").exists()      # upload removed after the job


def test_validation(client):
    assert submit(client, idx_name="movie.txt").status_code == 400
    assert submit(client, idx=b"nothing here").status_code == 400
    big = IDX + b"".join(f"timestamp: 00:01:{i % 60:02d}:000, filepos: 0\n".encode() for i in range(web.MAX_CUES))
    assert submit(client, idx=big).status_code == 413
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_job_rate_limit(client, monkeypatch):
    monkeypatch.setattr(web, "JOBS_PER_HOUR", 2)
    assert submit(client).status_code == 200
    assert submit(client).status_code == 200
    r = submit(client)
    assert r.status_code == 429 and "Retry-After" in r.headers


def test_daily_vlm_allowance_reduces_budget(client, monkeypatch):
    monkeypatch.setattr(web, "VLM_PER_DAY", 5)
    seen = []

    def spy(idx_path, config, progress):
        seen.append(config.max_vlm_cues)
        return fake_run_job_sync(idx_path, config, progress)
    monkeypatch.setattr(web.jobmod, "run_job_sync", spy)
    wait_done(client, submit(client).json()["id"])          # uses 3 of 5
    wait_done(client, submit(client).json()["id"])          # 2 left
    assert seen == [5, 2]


def test_stats_and_index(client):
    assert client.get("/healthz").json() == {"ok": True}
    s = client.get("/api/stats").json()
    assert "fonts" in s and s["limits"]["max_cues"] == web.MAX_CUES
    assert "<title>vobsub-to-srt" in client.get("/").text
