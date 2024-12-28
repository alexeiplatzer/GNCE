import json
import sys
import time
from tqdm import tqdm

from cardinality_estimation.constants import PROJECT_ROOT_PATH, DATASETS_PATH
from cardinality_estimation.GNCE.embeddings_generator import get_embeddings


if __name__ == "__main__":
    # Get entities from queries:
    entities = []
    dataset_name = "lubm"
    queries_file = DATASETS_PATH / dataset_name / "star" / "Joined_Queries.json"
    with open(queries_file, "r") as f:
        queries = json.load(f)
    for query in queries:
        entities += query["x"]

    entities = list(set(entities))
    print(entities)

    entities = entities[:]

    print("Using ", len(entities), " entities for RDF2Vec")

    print("Starting...")
    graph_file = DATASETS_PATH / dataset_name / "graph" / f"{dataset_name}.ttl"
    get_embeddings(
        dataset_name,
        graph_file,
        remote=True,
        entities=entities,
        sparql_endpoint="http://localhost:8909/sparql/",
    )
