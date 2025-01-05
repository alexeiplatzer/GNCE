import json
import sys
import time
from tqdm import tqdm

from ..constants import PROJECT_ROOT_PATH, DATASETS_PATH

from ..pyrdf2vec.graphs import KG
from ..pyrdf2vec import RDF2VecTransformer
from ..pyrdf2vec.embedders import Word2Vec
from ..pyrdf2vec.walkers import RandomWalker


def uri_to_id(uri):
    return uri.split("/")[-1]


"""
This file manages the generation of embeddings for the different entities

"""


uri_query = """
        PREFIX org: <https://w3id.org/scholarlydata/organisation/>

        SELECT *
        WHERE{{
        {{<{}> ?p ?o.}}
        UNION
        {{?s <{}> ?o.}}
        UNION
        {{?s ?p <{}>.}}

        }}
        LIMIT 30000000000
        """

literal_query = """
        PREFIX org: <https://w3id.org/scholarlydata/organisation/>

        SELECT *
        WHERE{{
        {{?s ?p '{}'.}}

        }}
        LIMIT 3000000000000
        """


def get_embeddings(
    dataset_name,
    kg_file,
    entities=None,
    remote=True,
    sparql_endpoint="http://127.0.0.1:8902/sparql/",
    n_jobs=12,
):
    """
    This function calculates simple occurences as well as rdf2vec embeddings for a given rdf graph
    Saves the embeddings and occurrences in one json file per entity
    :param dataset_name: The name of the kg, used to save the statistics
    :param kg_file: path to the .ttl file of the kg
    :param entities: list of entities for which to calculate embeddings
    :param remote: flag whether a remote sparql endpoint will be used or the .ttl file in memory
    :param sparql_endpoint: url of the remote sparql endpoint, if used
    :return: None
    """
    # GRAPH = KG(kg_file, skip_verify=True)
    GRAPH = KG(sparql_endpoint, skip_verify=True)

    # Dict to hold several runtimes
    timing_dict = {}

    if not remote and entities is None:
        train_entities = [entity.name for entity in list(GRAPH._entities)]
        test_entities = [entity.name for entity in list(GRAPH._vertices)]
        entities = set(train_entities + test_entities)

    # Create the RDF2vec model
    transformer = RDF2VecTransformer(
        Word2Vec(epochs=10, vector_size=100),
        walkers=[RandomWalker(4, max_walks=5, with_reverse=True, n_jobs=n_jobs, md5_bytes=None)],
        verbose=2,
        batch_mode="onefile",
    )

    # Count occurrences of nodes
    occurrences = {}

    elements_to_remove = []
    print("Calculating Occurences")
    occurrence_from_file = True

    # Start Time of Occurrence Calculation:
    occurrence_start_time = time.time()

    if occurrence_from_file:

        with open(kg_file, "r") as file:
            for line in tqdm(file):
                line = line.strip().split(
                    " "
                )  # Assuming the elements are separated by a space
                s = line[0].replace("<", "").replace(">", "")
                p = line[1].replace("<", "").replace(">", "")
                o = line[2].replace("<", "").replace(">", "")

                # Using dict.get() method to eliminate try-except blocks
                occurrences[s] = occurrences.get(s, 0) + 1
                occurrences[p] = occurrences.get(p, 0) + 1
                occurrences[o] = occurrences.get(o, 0) + 1

    else:
        raise NotImplementedError

    # Saving occurrences
    with open(dataset_name + "_ocurrences.json", "w") as fp:
        json.dump(occurrences, fp)

    # End Time of Occurrence Calculation
    occurrence_end_time = (time.time() - occurrence_start_time) * 1000
    n_atoms_occurrence = len(occurrences)
    time_per_atom_occurrence = occurrence_end_time / n_atoms_occurrence
    timing_dict["occurrence_total_time"] = occurrence_end_time
    timing_dict["n_atoms_occurrence"] = n_atoms_occurrence
    timing_dict["time_per_atom_occurrence"] = time_per_atom_occurrence

    # Deleting old walks:
    import os
    import shutil

    # Define the path of the walk folder
    folder_path = PROJECT_ROOT_PATH / "walks"
    # Delete the folder if it exists
    if folder_path.exists():
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    # Create the folder again
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Recreated folder: {folder_path}")

    # Timing for embedding calculation
    embedding_start_time = time.time()

    # Generate the embeddings
    print("Starting to fit model")
    # embeddings, literals = transformer.fit(GRAPH, entities)
    transformer.fit(GRAPH, entities)
    print("Finished fitting model")

    # Generating embedding for all entities
    test_entities_cleaned = []
    embeddings_test = []
    occurences_test = []

    i = -1
    print("Calculating Embeddings")
    for entity in tqdm(entities):
        i += 1
        try:
            embedding, literals = transformer.transform(GRAPH, [uri_to_id(entity)])
            test_entities_cleaned.append(entity)
            embeddings_test += embedding
            try:
                occurences_test.append(occurrences[entity])
            except KeyError:
                print("cant find occurrence for ", entity)
                occurences_test.append(0)
        except Exception as e:
            print(e)
            print(entity)
            raise

    print(len(occurrences))
    print(len(test_entities_cleaned))

    embedding_end_time = (time.time() - embedding_start_time) * 1000
    n_atoms_embeddings = len(test_entities_cleaned)
    time_per_atom_embedding = embedding_end_time / n_atoms_embeddings
    timing_dict["embedding_total_time"] = embedding_end_time
    timing_dict["n_atoms_embeddings"] = n_atoms_embeddings
    timing_dict["time_per_atom_embedding"] = time_per_atom_embedding
    timing_dict["time_per_atom_statistic"] = (
        time_per_atom_occurrence + time_per_atom_embedding
    )
    embedding_timing_file = DATASETS_PATH / dataset_name / "embedding_timing.json"
    with open(embedding_timing_file, "w") as fp:
        json.dump(timing_dict, fp, indent=4)

    # Storing embeddings one by one to separate files(necessary for large KG)
    print("Saving statistics")
    for i in tqdm(range(len(test_entities_cleaned))):
        statistics_dict = {
            "embedding": embeddings_test[i].tolist(),
            "occurence": occurences_test[i],
        }

        file_name = test_entities_cleaned[i].replace("/", "|")
        file_path = DATASETS_PATH / dataset_name / "statistics" / f"{file_name}.json"
        with open(file_path, "w") as fp:
            json.dump(statistics_dict, fp)
    return
