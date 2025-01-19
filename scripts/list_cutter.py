import json
from dataclasses import dataclass

from cardinality_estimation.constants import DATASETS_PATH


@dataclass
class Config:
    dataset_name = "lubm"
    query_type = "star"
    query_filename = "Joined_Queries.json"


def print_queries(start_idx, end_idx):
    config = Config()
    with open(
        DATASETS_PATH / config.dataset_name / config.query_type / config.query_filename, "r"
    ) as f:
        data = json.load(f)
    print(json.dumps(data[start_idx:end_idx], indent=4))


def print_graph(end_idx):
    config = Config()
    with open(
        DATASETS_PATH / config.dataset_name / "graph" / f"{config.dataset_name}.nt", "r"
    ) as f:
        idx = 0
        for line in f:
            if idx >= end_idx:
                break
            print(line.strip())
            idx += 1


if __name__ == "__main__":
    # print_queries(start_idx=0, end_idx=1)
    print_graph(end_idx=5)
