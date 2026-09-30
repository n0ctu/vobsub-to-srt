import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from vobsub_to_srt import web  # noqa: E402
from vobsub_to_srt.job import JobResult  # noqa: E402

IDX = b"# VobSub index\nsize: 720x576\nid: en, index: 0\n" + b"".join(
    f"timestamp: 00:00:{i:02d}:000, filepos: {i:09x}\n".encode() for i in range(5))


def fake_run_job_sync(source, config, progress):
    assert source.idx.startswith(b"# VobSub index") and isinstance(source.sub, bytes)   # in memory, no path
    progress({"event": "probe", "db": "calm-sable-0000", "mode": "exact", "coverage": 1.0})
    progress({"event": "cues", "items": [{"i": 0, "start": 0, "end": 1000, "text": "Hello", "src": "nocr"}], "resolved": 1})
    progress({"event": "glyphs", "items": [{"key": "k1", "label": "H", "w": 4, "h": 5, "bits": "8A==", "n": 3,
                                            "style": "", "new": False}], "known": 1, "pending": 0})
    progress({"event": "done", "unresolved": 0, "seconds": 0.1})
    return JobResult(srt="1\n00:00:00,000 --> 00:00:01,000\nHello\n",
                     report={"by_source": {"nocr": 4, "vlm": 1}, "cues": 5, "vlm_requests": 3, "seconds": 0.1, "language": "en"},
                     unresolved=0, vlm_used=3, budget_exhausted=False)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "DATA", tmp_path)
    monkeypatch.setattr(web.jobmod, "run_job_sync", fake_run_job_sync)
    web.jobs.clear()
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
    assert [e["event"] for e in events] == ["queued", "started", "probe", "cues", "glyphs", "done"]
    assert events[3]["items"][0]["text"] == "Hello" and events[4]["items"][0]["label"] == "H"
    assert all("db" not in e and "srt" not in e for e in events)          # nothing internal leaks
    assert web.jobs[job_id].upload is None                                # upload dropped after the job
    assert not (web.DATA / "jobs").exists()                                # nothing on disk


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

    def spy(source, config, progress):
        seen.append(config.max_vlm_cues)
        return fake_run_job_sync(source, config, progress)
    monkeypatch.setattr(web.jobmod, "run_job_sync", spy)
    wait_done(client, submit(client).json()["id"])          # uses 3 of 5
    wait_done(client, submit(client).json()["id"])          # 2 left
    assert seen == [5, 2]


def test_stats_and_index(client):
    assert client.get("/healthz").json() == {"ok": True}
    s = client.get("/api/stats").json()
    assert "fonts" in s and s["limits"]["max_cues"] == web.MAX_CUES
    assert "<title>VobSub to SRT Tool" in client.get("/").text


def test_static_files(client):
    r = client.get("/static/app.css")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/css")
    assert client.get("/static/index.html").status_code == 404          # only css/js/svg
    assert client.get("/static/..%2Fweb.py").status_code == 404


def test_queue_cap(client, monkeypatch):
    monkeypatch.setattr(web, "MAX_QUEUE", 0)
    r = submit(client)
    assert r.status_code == 503 and "Retry-After" in r.headers


def test_usage_stats_count_without_storing_users(client):
    for _ in range(2):
        wait_done(client, submit(client).json()["id"])
    u = client.get("/api/stats").json()["usage"]
    assert u["jobs"] == 2 and u["cues"] == 10 and u["memory_cues"] == 8 and u["vision_cues"] == 2
    assert u["vlm_requests"] == 6 and u["languages"] == {"en": 2} and u["users"] == 1   # same address, one user
    assert (web.DATA / "stats.sqlite").exists()
    assert b"testclient" not in (web.DATA / "stats.sqlite").read_bytes()          # only salted hashes
    again = web.Store(web.DATA / "stats.sqlite").snapshot()                        # survives a restart
    assert again["jobs"] == 2 and again["users"] == 1
    g = client.get("/api/stats").json()["glyphs"]
    assert set(g) == {"fonts", "fonts_learned_here", "shapes", "italic_shapes", "fused_shapes"}


def test_limits_persist_across_restart(client, monkeypatch):
    monkeypatch.setattr(web, "JOBS_PER_HOUR", 1)
    assert submit(client).status_code == 200
    fresh = web.Store(web.DATA / "stats.sqlite")                                   # a new process
    assert fresh.check_job("testclient") is not None                              # still rate limited
    assert fresh.vlm_left("testclient") == web.VLM_PER_DAY - 3 or fresh.vlm_left("testclient") == web.VLM_PER_DAY
