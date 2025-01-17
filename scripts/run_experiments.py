import time
import json
import os
import sys
from datetime import datetime

from cardinality_estimation.constants import PACKAGE_PATH
from cardinality_estimation.constants import DATASETS_PATH
from cardinality_estimation.GNCE.cardinality_estimation import train_GNCE
# from cardinality_estimation.LMKG.lmkgs.lmkgs import run_lmkg
# from cardinality_estimation.GCARE.run_estimation import run_gcare

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# os.chdir(GNCE_PATH + '/LMKG/lmkgs')
# sys.path.append(GNCE_PATH + '/LMKG/lmkgs')
# sys.path.append(GNCE_PATH + '/GCARE')


starttime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

query_filename = "Joined_Queries.json"

dataset = "lubm"
query_type = "star"

run_LMKG = False
run_GNCE = True
run_GCARE = False

# Whether to perform full inductive training
# Choices are 'false' or 'full'. false means normal training and full means evaluating without embeddings
inductive = "false"


# if run_LMKG:
#     print("**** Starting LMKG ****")
#     os.chdir(PACKAGE_PATH / "LMKG" / "lmkgs")
#     starting_time_lmkg = time.time()
#     # How many atoms will be trained and evaluated on
#     n_atoms_lmkg = 0
#     n_atoms_lmkg += run_lmkg(
#         dataset=dataset,
#         query_form=query_type,
#         eval_folder=starttime,
#         query_filename=query_filename,
#         train=True,
#         inductive=inductive,
#         DATASETPATH=DATASETS_PATH,
#     )
#     n_atoms_lmkg += run_lmkg(
#         dataset=dataset,
#         query_form=query_type,
#         eval_folder=starttime,
#         query_filename=query_filename,
#         train=False,
#         inductive=inductive,
#         DATASETPATH=DATASETS_PATH,
#     )
#     # How long training and evaluating takes per atom in ms
#     total_training_time_per_atom = (
#         (time.time() - starting_time_lmkg) / n_atoms_lmkg * 1000
#     )
#     print(f"Training LMKG took {total_training_time_per_atom} ms per atom")
#     print(f"Trained on a total of {n_atoms_lmkg} token")
#
#     training_timing = {"total_training_time_per_atom": total_training_time_per_atom}
#     with open(
#         f"{DATASETS_PATH}/{dataset}/Results/{starttime}/LMKG/training_timing.json", "w"
#     ) as file:
#         json.dump(training_timing, file, indent=4)


if run_GNCE:
    print("**** Starting GNCE ****")
    os.chdir(PACKAGE_PATH / "GNCE")
    start = time.time()
    n_atoms, start_time_gnce, end_time_gnce = train_GNCE(
        dataset=dataset,
        query_type=query_type,
        query_filename=query_filename,
        eval_folder=starttime,
        inductive=inductive,
        DATASETPATH=DATASETS_PATH,
    )

    end = time.time()

    # Note: start_time_gnce and end_time_gnce are the times for only the training loop, without data loading
    total_training_time_per_atom = (end - start) / n_atoms * 1000
    print(f"Training GNCE took {total_training_time_per_atom} ms per atom")
    print(f"Trained on a total of {n_atoms} token")
    training_timing = {
        "total_training_time_per_atom": total_training_time_per_atom,
        "n_atoms": n_atoms,
        "total_time": (end_time_gnce - start_time_gnce) * 1000,
    }
    with open(
        DATASETS_PATH / dataset / f"Results/{starttime}/GNCE/training_timing.json", "w"
    ) as file:
        json.dump(training_timing, file, indent=4)


# if run_GCARE:
#     print("**** Starting GCARE ****")
#     os.chdir(PACKAGE_PATH / "GCARE")
#     run_gcare(
#         dataset=dataset,
#         query_type=query_type,
#         eval_folder=starttime,
#         query_filename=query_filename,
#         inductive=inductive,
#     )
