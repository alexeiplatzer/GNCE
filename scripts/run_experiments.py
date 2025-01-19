import time
import json
import os
import sys
from datetime import datetime
from dataclasses import dataclass

from cardinality_estimation.constants import PACKAGE_PATH
from cardinality_estimation.constants import DATASETS_PATH
from cardinality_estimation.GNCE.cardinality_estimation import train_GNCE


@dataclass
class Config:
    dataset = "lubm"
    query_type = "star"
    query_filename = "Joined_Queries.json"

    total_data_count = 1000
    train_data_count = 800

    # Whether to perform full inductive training
    # Choices are 'false' or 'full'.
    # false means normal training and full means evaluating without embeddings
    inductive = "false"


if __name__ == "__main__":
    config = Config()
    print("**** Starting GNCE ****")
    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.chdir(PACKAGE_PATH / "GNCE")
    start = time.time()
    n_atoms, start_time_gnce, end_time_gnce = train_GNCE(
        dataset_name=config.dataset,
        query_type=config.query_type,
        query_filename=config.query_filename,
        eval_folder=start_time,
        DATASET_PATH=DATASETS_PATH,
        total_data_count=config.total_data_count,
        train_data_count=config.train_data_count,
    )

    end = time.time()

    # Note: start_time_gnce and end_time_gnce are the times for only the training loop,
    # without data loading
    total_training_time_per_atom = (end - start) / n_atoms * 1000
    print(f"Training GNCE took {total_training_time_per_atom} ms per atom")
    print(f"Trained on a total of {n_atoms} token")
    training_timing = {
        "total_training_time_per_atom": total_training_time_per_atom,
        "n_atoms": n_atoms,
        "total_time": (end_time_gnce - start_time_gnce) * 1000,
    }
    with open(
        DATASETS_PATH / config.dataset / f"Results/{start_time}/GNCE/training_timing.json", "w"
    ) as file:
        json.dump(training_timing, file, indent=4)
