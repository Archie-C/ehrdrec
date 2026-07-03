import sys
from types import SimpleNamespace

from ehrdrec.training.logging import WandbLogger


class FakeRun:
    def __init__(self) -> None:
        self.logged = []
        self.summary = {}
        self.finished = False

    def log(self, payload, step=None) -> None:
        self.logged.append((payload, step))

    def finish(self) -> None:
        self.finished = True


def test_wandb_logger_logs_metrics_and_best_summary(monkeypatch):
    fake_run = FakeRun()

    def init(**kwargs):
        fake_run.init_kwargs = kwargs
        return fake_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))

    logger = WandbLogger(
        project="ehrdrec",
        name="unit-test",
        config={"lr": 1e-3},
        mode="disabled",
    )

    logger.log({"custom_metric": 1.2}, step=2)
    logger.on_epoch_end(
        3,
        train_metrics={"Jaccard": 0.2, "Binary DDI": 0.1},
        val_metrics={"1bad name": 0.4},
    )
    logger.on_best_model(3, 0.4, state_dict={})
    logger.close()

    assert fake_run.init_kwargs["project"] == "ehrdrec"
    assert fake_run.init_kwargs["name"] == "unit-test"
    assert fake_run.init_kwargs["config"] == {"lr": 1e-3}
    assert fake_run.init_kwargs["mode"] == "disabled"
    assert fake_run.logged[0] == ({"custom_metric": 1.2}, 2)
    assert fake_run.logged[1] == (
        {
            "epoch": 3,
            "train_Jaccard": 0.2,
            "train_Binary_DDI": 0.1,
            "val_metric_1bad_name": 0.4,
        },
        3,
    )
    assert fake_run.logged[2] == ({"best_epoch": 3, "best_score": 0.4}, 3)
    assert fake_run.summary == {"best_epoch": 3, "best_score": 0.4}
    assert fake_run.finished
