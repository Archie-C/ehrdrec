from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib.metadata
import inspect
import os
from pathlib import Path
import platform
import random
import sys
from typing import Any

import torch

from ehrdrec.contracts.experiment_output import (
    DataSplitInformation,
    DatasetInformation,
    HardwareInformation,
    ModelInformation,
    PackageInformation,
    SoftwareEnvironment,
    SourceFile,
)
from ehrdrec.data.torch import EHRDataset
from ehrdrec.experiments.artifacts import (
    sha256_text,
    stable_fingerprint,
    to_jsonable,
)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed installed RNG backends and request deterministic PyTorch behavior."""

    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def seed_worker(_: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(worker_seed)


def capture_source_file(value: object) -> SourceFile:
    """Capture inspectable source without making source availability fatal."""

    filename = "<source unavailable>"
    content = ""

    try:
        source_path = inspect.getsourcefile(value)
        if source_path is not None:
            path = Path(source_path).resolve()
            filename = str(path)
            content = path.read_text(encoding="utf-8")
        else:
            content = inspect.getsource(value)
            filename = f"<{type(value).__name__}>"
    except (OSError, TypeError):
        try:
            content = inspect.getsource(value)
            filename = f"<{getattr(value, '__name__', type(value).__name__)}>"
        except (OSError, TypeError):
            pass

    return SourceFile(
        filename=filename,
        sha256=sha256_text(content),
        content=content,
    )


def resolve_callable_config(
    factory: Callable[..., object],
    supplied: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve explicit and default constructor settings, excluding context."""

    supplied = dict(supplied or {})
    resolved: dict[str, Any] = {}

    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return to_jsonable(supplied)

    for name, parameter in parameters.items():
        if name in {"self", "context"}:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if name in supplied:
            resolved[name] = supplied[name]
        elif parameter.default is not inspect.Parameter.empty:
            resolved[name] = parameter.default

    for name, value in supplied.items():
        resolved.setdefault(name, value)

    return to_jsonable(resolved)


def capture_model_information(
    model_factory: Callable[..., object],
    resolved_config: Mapping[str, Any],
    config_source: str | Path | None = None,
) -> ModelInformation:
    source = capture_source_file(model_factory)
    captured_config = None

    if config_source is not None:
        path = Path(config_source).resolve()
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            content = ""
        captured_config = SourceFile(
            filename=str(path),
            sha256=sha256_text(content),
            content=content,
        )

    return ModelInformation(
        name=getattr(model_factory, "__name__", type(model_factory).__name__),
        source=source,
        resolved_config=dict(to_jsonable(resolved_config)),
        config_source=captured_config,
    )


def capture_software_environment() -> SoftwareEnvironment:
    packages: dict[str, str] = {}

    for distribution in importlib.metadata.distributions():
        try:
            name = distribution.metadata.get("Name")
            version = distribution.version
            if name and version:
                packages[name] = version
        except Exception:
            continue

    try:
        ehrdrec_version = importlib.metadata.version("ehrdrec")
    except importlib.metadata.PackageNotFoundError:
        ehrdrec_version = "unknown"

    return SoftwareEnvironment(
        python_version=platform.python_version(),
        ehrdrec_version=ehrdrec_version,
        packages=[
            PackageInformation(name=name, version=packages[name])
            for name in sorted(packages, key=str.casefold)
        ],
        operating_system=platform.platform(),
    )


def capture_hardware_information() -> HardwareInformation:
    cpu = platform.processor() or platform.machine() or None

    try:
        for line in Path("/proc/cpuinfo").read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    ram_gb = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        ram_gb = round(page_size * page_count / 1024**3, 3)
    except (OSError, ValueError):
        pass

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu = None
    if gpu_count:
        try:
            gpu = ", ".join(
                torch.cuda.get_device_name(index)
                for index in range(gpu_count)
            )
        except (AssertionError, RuntimeError):
            gpu = None

    cudnn = torch.backends.cudnn.version()

    return HardwareInformation(
        cpu=cpu,
        cpu_count=os.cpu_count(),
        ram_gb=ram_gb,
        gpu=gpu,
        gpu_count=gpu_count,
        cuda_version=torch.version.cuda,
        cudnn_version=str(cudnn) if cudnn is not None else None,
    )


def resolved_task_settings(task: object) -> dict[str, Any]:
    resolver = getattr(task, "get_resolved_config", None)
    if callable(resolver):
        settings = resolver()
    else:
        settings = getattr(task, "config", {})
    return dict(to_jsonable(settings))


def capture_dataset_information(
    datasets: Mapping[str, EHRDataset],
    name: str,
    version: str | None,
    sources: list[str],
) -> DatasetInformation:
    splits: list[DataSplitInformation] = []
    all_patients: set[str] = set()
    all_visits: set[str] = set()

    for split_name in ("train", "validation", "test"):
        dataset = datasets[split_name]
        frame = dataset.frame
        columns = set(frame.columns)

        patients = (
            {str(value) for value in frame["SUBJECT_ID"].to_list()}
            if "SUBJECT_ID" in columns
            else set()
        )
        visits = (
            {
                f"{subject}:{visit}"
                for subject, visit in frame.select(
                    "SUBJECT_ID", "HADM_ID"
                ).iter_rows()
            }
            if {"SUBJECT_ID", "HADM_ID"} <= columns
            else (
                {str(value) for value in frame["HADM_ID"].to_list()}
                if "HADM_ID" in columns
                else set()
            )
        )

        if "EXAMPLE_ID" in columns:
            identifiers = [str(value) for value in frame["EXAMPLE_ID"].to_list()]
        elif {"SUBJECT_ID", "HADM_ID"} <= columns:
            identifiers = [
                f"{subject}:{visit}"
                for subject, visit in frame.select(
                    "SUBJECT_ID", "HADM_ID"
                ).iter_rows()
            ]
        else:
            identifiers = [
                f"{split_name}:{index:08d}"
                for index in range(len(dataset))
            ]

        all_patients.update(patients)
        all_visits.update(visits)
        splits.append(
            DataSplitInformation(
                name=split_name,
                num_examples=len(dataset),
                num_patients=len(patients) if patients else None,
                num_visits=len(visits) if visits else None,
                fingerprint=stable_fingerprint(
                    {
                        "split": split_name,
                        "example_identifiers": identifiers,
                    }
                ),
            )
        )

    return DatasetInformation(
        name=name,
        version=version,
        sources=sorted(sources),
        fingerprint=stable_fingerprint(
            {
                split.name: split.fingerprint
                for split in splits
            }
        ),
        num_patients=len(all_patients) if all_patients else None,
        num_visits=len(all_visits) if all_visits else None,
        num_examples=sum(split.num_examples for split in splits),
        splits=splits,
    )


def current_command() -> str | None:
    if not sys.argv:
        return None
    return " ".join(sys.argv)
