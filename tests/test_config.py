import pytest

from src.utils.config import load_yaml, get_pipeline_paths


def test_load_yaml_requires_mapping(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")  # YAML list, not dict
    with pytest.raises(ValueError):
        load_yaml(p)


def test_get_pipeline_paths_requires_storage_keys():
    with pytest.raises(ValueError):
        get_pipeline_paths({"storage": {"processed_dir": "x"}})
