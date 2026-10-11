import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def trajectory_entry():
    spec = importlib.util.spec_from_file_location(
        "trajectory_training_schedule", ROOT / "runs/trajectory/train_trajectory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cosine_schedule_keeps_initial_rate_and_reaches_minimum_in_epoch_65():
    module = trajectory_entry()
    config = json.loads((ROOT / "configs/retrain/s43/trajectory/T1.json").read_text())
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=config["learning_rate"])
    scheduler = module._build_lr_scheduler(optimizer, config, 65)
    rates = []
    for _ in range(65):
        rates.append(optimizer.param_groups[0]["lr"])
        parameter.square().sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    assert rates[:10] == pytest.approx([2e-4] * 10)
    assert 2e-5 < rates[10] < rates[9]
    assert all(later < earlier for earlier, later in zip(rates[9:], rates[10:]))
    assert rates[-1] == pytest.approx(2e-5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)


def test_legacy_config_without_schedule_keeps_fixed_learning_rate():
    module = trajectory_entry()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=2e-4)
    assert module._build_lr_scheduler(optimizer, {"learning_rate": 2e-4}, 60) is None
    assert optimizer.param_groups[0]["lr"] == 2e-4


@pytest.mark.parametrize("group", ["T1", "T2", "T3"])
@pytest.mark.parametrize("budget", [2, 12])
def test_training_entry_records_used_rates_in_both_stages(tmp_path, monkeypatch, group, budget):
    module = trajectory_entry()
    config = json.loads((ROOT / f"configs/retrain/s43/trajectory/{group}.json").read_text())
    config.pop("protocol")
    config["device"] = "cpu"
    if group == "T3":
        config["mode"] = "two_stage"
        config["pretrain_epochs"] = 10
        config["finetune_epochs"] = 2
    source = tmp_path / "config.json"
    source.write_text(json.dumps(config))
    run = tmp_path / "run"
    dataset = torch.utils.data.TensorDataset(torch.ones(2, 1))
    dataset.manifest = {}
    dataset.scenes = [SimpleNamespace(scene_path=f"/data/task_0001/scene_{i}")
                      for i in range(2)]
    mapping = {"train": ["task_0001/scene_1"], "val": [], "test": ["task_0001/scene_0"]}
    preserved_split = tmp_path / "saved_split.json"
    preserved_split.write_text(json.dumps(mapping))
    model = torch.nn.Linear(1, 1)
    observed, stages = [], []

    def train_epoch(model, loader, optimizer, device, *args, **kwargs):
        observed.append(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad()
        loss = model(torch.ones(1, 1)).square().mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def pretrain(*args, **kwargs):
        stages.append("pretrain")
        return train_epoch(*args, **kwargs)

    def augment(*args, **kwargs):
        stages.append("joint")
        loss = train_epoch(*args, **kwargs)
        return loss, loss, loss

    monkeypatch.setattr(module, "build_trajectory_dataset", lambda _: dataset)
    monkeypatch.setattr(module, "build_trajectory_model", lambda *args: model)
    monkeypatch.setattr(module, "train_trajectory_epoch", train_epoch)
    monkeypatch.setattr(module, "train_augmented_trajectory_epoch", augment)
    monkeypatch.setattr(module, "pretrain_intra_modal_epoch", pretrain)
    monkeypatch.setattr(sys, "argv", ["train_trajectory", "--config", str(source),
                                     "--run-dir", str(run), "--epochs", str(budget),
                                     "--split-manifest", str(preserved_split)])
    threads = torch.get_num_threads()
    try:
        module.main()
    finally:
        torch.set_num_threads(threads)
    history = json.loads((run / "curves.json").read_text())
    assert json.loads((run / "split_manifest.json").read_text()) == mapping
    assert len(history["train_loss"]) == len(history["learning_rate"]) == budget
    assert history["learning_rate"] == pytest.approx(observed)
    assert observed[:min(10, budget)] == pytest.approx([2e-4] * min(10, budget))
    if budget > 10:
        assert observed[-1] == pytest.approx(2e-5)
    if group == "T3":
        pretrain_count = 1 if budget == 2 else 10
        assert stages == ["pretrain"] * pretrain_count + ["joint"] * (budget - pretrain_count)
        assert history["train_loss"][:pretrain_count] == [None] * pretrain_count
    assert (run / "best_model.pth").is_file()
    assert not (run / "last_model.pth").exists()


@pytest.mark.parametrize("losses,epoch", [([.9, .73, .72, .6], 3),
                                         ([.9, .8, .73, .74], None)])
def test_loss_checkpoint_saves_first_strict_threshold_crossing_only(
        tmp_path, monkeypatch, losses, epoch):
    module = trajectory_entry()
    config = json.loads((ROOT / "configs/retrain/s43/trajectory/T0.json").read_text())
    config["device"] = "cpu"
    config["num_epochs"] = len(losses)
    config["model_params"]["use_gripper"] = False
    source = tmp_path / "config.json"
    source.write_text(json.dumps(config))
    run = tmp_path / "run"
    dataset = torch.utils.data.TensorDataset(torch.ones(2, 1))
    dataset.manifest = {}
    dataset.scenes = [SimpleNamespace(scene_path=f"/data/task_0001/scene_{i}")
                      for i in range(2)]
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["task_0001/scene_0"], "val": [],
                                 "test": ["task_0001/scene_1"]}))
    model = torch.nn.Linear(1, 1)
    steps, saved = [], []
    save = torch.save

    def train(model, loader, optimizer, *args, **kwargs):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).square().mean().backward()
        optimizer.step()
        steps.append(len(steps) + 1)
        with torch.no_grad():
            model.weight.fill_(steps[-1])
        return losses[len(steps) - 1]

    def record_save(state, path):
        saved.append(Path(path).name)
        save(state, path)

    monkeypatch.setattr(module, "build_trajectory_dataset", lambda _: dataset)
    monkeypatch.setattr(module, "build_trajectory_model", lambda *args: model)
    monkeypatch.setattr(module, "train_trajectory_epoch", train)
    monkeypatch.setattr(module.torch, "save", record_save)
    monkeypatch.setattr("bise.common.evaluation.bind_protocols", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["train_trajectory", "--config", str(source),
                                     "--run-dir", str(run), "--split-manifest", str(split)])
    threads = torch.get_num_threads()
    try:
        module.main()
    finally:
        torch.set_num_threads(threads)
    assert (run / "best_model.pth").is_file()
    assert not (run / "last_model.pth").exists()
    status = json.loads((run / "status.json").read_text())
    if epoch is None:
        assert "loss_model.pth" not in saved
        assert not (run / "loss_model.pth").exists()
        assert "loss_checkpoint" not in status
    else:
        assert saved.count("loss_model.pth") == 1
        assert status["loss_checkpoint"] == {"epoch": epoch, "train_loss": .72,
                                             "threshold": .73}
        state = torch.load(run / "loss_model.pth", weights_only=True)
        torch.testing.assert_close(state["weight"], torch.full_like(state["weight"], epoch))
