"""Verify portable artifact archives without model weights or source mutations."""

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pack_retrain", Path(__file__).resolve().parents[1] / "scripts/pack_retrain.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_archive_keeps_raw_comparison_data_and_excludes_all_weight_variants(tmp_path):
    source = tmp_path / "retrain"
    run = source / "s44/video/E5"
    run.mkdir(parents=True)
    keep = ("params.json", "split_manifest.json", "metrics.json", "train.log", "chart.png",
            "human_embeddings.npy", "similarity_matrix.npy", "semantic_teachers.npz")
    weights = ("best_model.pth", "best_model_origin.pth", "loss_model.pth", "backbone.pt",
               "model.CKPT", "model.safetensors", "pytorch_model.bin", "model.onnx",
               "checkpoint.pth.tar", "checkpoint.pt.tar", "checkpoint.ckpt.tar")
    for name in keep + weights:
        (run / name).write_bytes(name.encode())
    report = MODULE.pack_retrain(source)
    assert report["files"] == len(keep) and report["excluded_weights"] == len(weights)
    output = tmp_path / "export.zip"
    MODULE.pack_retrain(source, output)
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {f"retrain/s44/video/E5/{name}" for name in keep}
        for name in keep:
            assert archive.read(f"retrain/s44/video/E5/{name}") == name.encode()
    assert all((run / name).read_bytes() == name.encode() for name in keep + weights)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        MODULE.pack_retrain(source, output)
    assert output.read_bytes() == before


def test_archive_cannot_include_itself_or_follow_external_links(tmp_path):
    source = tmp_path / "retrain"
    source.mkdir()
    (source / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="outside"):
        MODULE.pack_retrain(source, source / "export.zip")
    external = tmp_path / "model.pth"
    external.write_bytes(b"external weights")
    (source / "disguised.json").symlink_to(external)
    with pytest.raises(ValueError, match="Symbolic links"):
        MODULE.pack_retrain(source, tmp_path / "export.zip")
