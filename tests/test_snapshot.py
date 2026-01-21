import json
from pathlib import Path

from src.utils.snapshot import save_text_snapshot, append_manifest_jsonl


def test_save_text_snapshot_writes_and_hashes(tmp_path):
    rec = save_text_snapshot(
        text="<html><body>Hello</body></html>",
        out_dir=tmp_path,
        prefix="t",
        url="https://example.test/x",
    )

    p = Path(rec.path)
    assert p.exists()
    b = p.read_bytes()

    import hashlib
    assert rec.sha256 == hashlib.sha256(b).hexdigest()
    assert rec.bytes == len(b)
    assert rec.url == "https://example.test/x"


def test_append_manifest_jsonl_appends_valid_json(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rec = save_text_snapshot(
        text="Hi",
        out_dir=tmp_path,
        prefix="t",
        url="https://example.test/y",
    )
    append_manifest_jsonl(manifest, rec)

    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["url"] == "https://example.test/y"
    assert "sha256" in obj
