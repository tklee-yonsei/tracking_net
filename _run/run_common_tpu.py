import argparse
import os
import subprocess
from typing import List, Optional, TypeVar

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def check_both_exists_or_not(a: Optional[T1], b: Optional[T2]) -> bool:
    return not (any([a, b]) and not all([a, b]))


def check_all_exists_or_not(list: List[T1]) -> bool:
    return not (any(list) and not all(list))


def loss_coords(s):
    try:
        x, y = s.split(",")
        return (x, float(y))
    except:
        raise argparse.ArgumentTypeError("Loss must be x,y")


def setup_continuous_training(
    continuous_model_name: Optional[str], continuous_epoch: Optional[int]
) -> Optional[str]:
    training_id: Optional[str] = None
    # extract `training_id` from `continuous_model_name`
    if continuous_model_name is not None:
        continuous_run_id: str = os.path.basename(continuous_model_name)
        if continuous_run_id.find(".") != -1:
            continuous_run_id = continuous_run_id[: continuous_run_id.find(".")]
        training_id = continuous_run_id
    return training_id


def create_tpu(
    tpu_name: str,
    ctpu_zone: str,
    # range: str = "10.240.0.0/29",
    accelerator_type: str = "v3-8",
    version="2.3.1",
):
    subprocess.run(
        [
            "gcloud",
            "compute",
            "tpus",
            "create",
            tpu_name,
            "--zone",
            ctpu_zone,
            # "--range",
            # range,
            "--accelerator-type",
            accelerator_type,
            "--version",
            version,
            "--preemptible",
            "--quiet",
        ]
    )


def delete_tpu(tpu_name: str, ctpu_zone: str):
    subprocess.run(
        [
            "gcloud",
            "compute",
            "tpus",
            "delete",
            tpu_name,
            "--zone",
            ctpu_zone,
            "--quiet",
        ]
    )
