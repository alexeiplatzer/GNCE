import json
import os
import time
import random
from datetime import datetime
from pathlib import Path

import git
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss

from .models import TripleModel
from .utils import StatisticsLoader, get_query_graph_data_new, ToUndirectedCustom
from ..LMKG.lmkgs.complex_reader import star_to_triples


class Q_Error(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        return torch.max(x / (y + self.epsilon), y / (x + self.epsilon))


def preprocess_dataset(dataset, statistics, n_atoms, device="cpu"):
    # Preparing dataset
    processed_dataset = []
    for datapoint in tqdm(dataset):
        # Get graph representation of query
        data, n_atoms = get_query_graph_data_new(datapoint, statistics, device, n_atoms=n_atoms)

        # Transform graph to undirected representation,
        # with feature indicating edge direction
        data = ToUndirectedCustom(merge=False)(data)
        data = data.to_homogeneous()
        data = data.to(device)
        y = np.log(datapoint["y"])
        y = torch.tensor(y)
        processed_dataset.append((data, y))
    return processed_dataset, n_atoms


def evaluate_GNN(
    test_data,
    results_path,
    statistics,
    device="cpu",
):
    # Evaluation of the best model on the test set
    model = TripleModel().to(device).double()
    model.load_state_dict(torch.load(results_path / "model.pth"))

    abs_errors = []
    q_errors = []
    preds = []
    gts = []
    sizes = []

    result_data = []

    model.eval()
    # List to store execution times
    exec_times = []
    exec_times_total = []
    for datapoint in test_data:
        start = time.time()
        data = get_query_graph_data_new(datapoint, statistics, device)
        data = ToUndirectedCustom(merge=False)(data)
        data = data.to_homogeneous()
        data = data.to(device)

        # Measure execution time of model
        start2 = time.time()
        out = model(
            data.x.double(),
            data.edge_index,
            data.edge_type,
            data.edge_attr.double(),
        )
        end = time.time()
        exec_times.append((end - start2) * 1000)  # Convert to ms
        exec_times_total.append((end - start) * 1000)
        sizes.append(len(datapoint["triples"]))
        pred = out.detach().cpu().numpy()[0][0]

        y = datapoint["y"]
        pred = np.exp(pred)

        # Storing results to np arrays and full result dict:
        preds.append(pred)
        gts.append(y)
        datapoint["y_pred"] = pred
        datapoint["exec_time"] = (end - start2) * 1000
        datapoint["exec_time_total"] = (end - start) * 1000

        result_data.append(datapoint)
        y = torch.tensor(y).double()
        abs_errors.append(np.abs(pred - y))
        q_errors.append(np.max([np.abs(pred) / y, y / np.abs(pred)]))

    print("Mean Absolute Error: ", np.mean(abs_errors))
    print("Mean Q-Error: ", np.mean(q_errors))
    print("Mean execution time: ", np.mean(exec_times))
    print("Mean execution time total: ", np.mean(exec_times_total))

    np.save(results_path / "preds.npy", preds)
    np.save(results_path / "gts.npy", gts)
    np.save(results_path / "sizes.npy", sizes)
    np.save(results_path / "pred_times.npy", exec_times)
    np.save(results_path / "pred_times_total.npy", exec_times_total)

    with open(results_path / "results.json", "w") as file:
        json.dump(result_data, file, indent=4)


def train_GNN(
    train_data,
    test_data,
    results_path,
    statistics,
    preparation_time: float,
    epochs=100,
    train=True,
    inductive="false",
    batch_size=32,
    DATASET_PATH=None,
    device="cpu",
):
    """
    Train the model on the given train_data, or evaluate on the given test_data
    :param train_data: training data in the form of a list of query dicts
    :param test_data: test data in the form of a list of query dicts
    :param results_path: path to save the results to
    :statistics: embeddings of the edges
    :param preparation_time: time in seconds it took to prepare the data before training
    :param epochs: number of epochs to train for
    :param train: if True, train the model, if False, evaluate the model
    :inductive Whether to train the model so that it can accept unknown entities ('true')
        and replaces all embeddings in testing to random(true)
        or whether to always use embeddings('false')
    :return: None
    """
    print("Starting Training...")
    test_mae = []
    test_q_error = []
    training_progress = []
    min_q_error = 9999999
    min_mae = 9999999
    # model = GINmodel().to(device).double()
    model = TripleModel().to(device).double()
    # Optionally, start from a checkpoint
    # try:
    #     model.load_state_dict(torch.load("model.pth"))
    # except Exception as e:
    #     print("No checkpoint found, starting with random weights")

    print("Number of Parameters: ", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss = MSELoss()
    # loss = Q_Error()

    # How many atoms are in total in the queries:
    n_atoms = 0

    # Preparing datasets
    start_time_preparation = time.time()

    # Preparing training sets
    X_train, n_atoms = preprocess_dataset(train_data, statistics, n_atoms, device)

    # Preparing test set
    X_test, n_atoms = preprocess_dataset(test_data, statistics, n_atoms, device)

    # Preparation Time per atom in ms
    preparation_time += time.time() - start_time_preparation
    preparation_time = preparation_time / n_atoms * 1000

    start_time_training = time.time()

    for epoch in tqdm(range(epochs)):
        start_time = time.time()

        epoch_loss = 0
        train_q_errors = []
        points_processed = 0
        i = 0

        model.train()
        # for datapoint in train_data:
        for data, y in tqdm(X_train):

            i += 1

            # Predict logarithm of cardinality
            out = model(
                data.x.double(),
                data.edge_index,
                data.edge_type,
                data.edge_attr.double(),
            )

            # Calculate loss
            l = loss(out, torch.tensor(y).to(device))

            l.backward()
            points_processed += 1
            # Gradient Accumulation
            if points_processed >= batch_size:
                optimizer.step()
                optimizer.zero_grad()
                points_processed = 0

            epoch_loss += l.item()
            pred = out.detach().cpu().numpy()[0][0]
            # As model predicts logarithm, scale accordingly
            pred = np.exp(pred)
            y = np.exp(y)
            train_q_errors.append(np.max([np.abs(pred) / y, y / np.abs(pred)]))

        print(
            f"Epoch {epoch}, Train Loss: {epoch_loss / len(train_data)}, "
            f"Avg Train Q-Error: {np.mean(train_q_errors)}"
        )

        # Evaluating on test set:
        abs_errors = []
        q_errors = []
        preds = []
        gts = []
        sizes = []

        model.eval()

        for data, y in X_test:

            out = model(
                data.x.double(),
                data.edge_index,
                data.edge_type,
                data.edge_attr.double(),
            )

            y = np.exp(y)

            pred = out.detach().cpu().numpy()[0][0]
            # As model predicts logarithm, scale accordingly
            pred = np.exp(pred)
            preds.append(pred)
            gts.append(y)
            abs_errors.append(np.abs(pred - y))
            q_errors.append(np.max([np.abs(pred) / y, y / np.abs(pred)]))
            sizes.append(len(datapoint["triples"]))

            points_processed += 1

        # Calculate mean absolute error and q-error
        print("MAE: ", np.mean(abs_errors))
        test_mae.append(np.mean(abs_errors))
        print("Qerror: ", np.mean(q_errors))
        test_q_error.append(np.mean(q_errors))

        end_time2 = time.time()
        epoch_time = end_time2 - start_time * 1000
        print("Time taken for one epoch:", epoch_time, "seconds")

        # Time per atom
        time_per_atom = epoch_time / n_atoms

        epoch_dict = {
            "epoch": epoch,
            "duration": epoch_time,
            "qerror": np.mean(q_errors),
            "mae": np.mean(abs_errors),
            "duration_per_atom": time_per_atom,
            "preparation_time_per_atom": preparation_time,
        }
        training_progress.append(epoch_dict)

        # Save model if it is the best so far
        if np.mean(q_errors) < min_q_error:
            print("New smallest Q-Error, saving model and statistics")
            torch.save(model.state_dict(), results_path / "model.pth")

            min_q_error = np.mean(q_errors)
            np.save(results_path / "preds.npy", preds)
            np.save(results_path / "gts.npy", gts)
            np.save(results_path / "sizes.npy", sizes)

        if np.mean(abs_errors) < min_mae:
            torch.save(model.state_dict(), results_path / "model_mae.pth")

            min_mae = np.mean(abs_errors)

    with open(results_path / "training_progress.json", "w") as file:
        json.dump(training_progress, file, indent=4)

    training_end_time = time.time()

    return n_atoms, start_time_training, training_end_time


def train_GNCE(
    dataset_name: str,
    query_type: str,
    eval_folder: str,
    query_filename: str,
    train: bool = True,
    DATASET_PATH: Path = None,
    total_data_count: int = 1000,
    train_data_count: int = 800,
):

    # Total counter for preparation, i.e. data loading and transforming to PyG graphs
    preparation_time = 0

    sim_measure = "cosine"

    # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"

    print("Using Device: ", device)

    # Load Statistics from disk
    statistics = StatisticsLoader(DATASET_PATH / dataset_name / "statistics")
    print("Successfully loaded statistics")

    eval_folder = Path(f"{eval_folder}/GNCE")

    start_time = time.time()

    with open(DATASET_PATH / dataset_name / query_type / query_filename, "r") as f:
        data = json.load(f)

    random.Random(4).shuffle(data)
    train_data_ratio = train_data_count / total_data_count
    test_data_count = total_data_count - train_data_count
    train_data = data[: int(train_data_ratio * len(data))][:train_data_count]
    test_data = data[int(train_data_ratio * len(data)) :][:test_data_count]

    preparation_time += time.time() - start_time

    print("Training on: ", len(train_data), " queries")
    print("Evaluating on: ", len(test_data), " queries")

    results_path = DATASET_PATH / dataset_name / "Results" / eval_folder

    # Create folder if it doesn't exist
    results_path.mkdir(parents=True, exist_ok=True)

    n_atoms, start_time_training, end_time_training = train_GNN(
        train_data,
        test_data,
        results_path=results_path,
        statistics=statistics,
        epochs=100,
        train=train,
        preparation_time=preparation_time,
        DATASET_PATH=DATASET_PATH,
        device=device,
    )

    return n_atoms, start_time_training, end_time_training
