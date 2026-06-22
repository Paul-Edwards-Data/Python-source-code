# ==============================================================================
# ATTRIBUTION & ACKNOWLEDGEMENT NOTICE
# This simulation framework was written/adapted by Paul Edwards.
#
# The implementation incorporates standard machine learning workflows and
# publicly available algorithmic logic (FedAvg, FedProx, Krum, and BadNets)
# adapted from their respective original research publications.
# All mathematical concepts and libraries utilized are credited to their
# respective authors via inline citations throughout this document.
# ==============================================================================

# Refeference: A. Yousefpour et al., "Opacus: User-friendly differential privacy in PyTorch,"
# arXiv preprint arXiv:2109.12298, 2021.
!pip install opacus

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator # Import ModuleValidator
from opacus.grad_sample import GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
import numpy as np
import copy
import pandas as pd # Import pandas for data saving
import sys # Import sys for command line arguments
from google.colab import files
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F # Added for softmax transformation

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*Secure RNG turned off.*",
)
warnings.filterwarnings(
    "ignore",
    message=".*PrivacyEngine detected new dataset object.*",
)

# --- Config and Hyperparameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIENTS_COUNT = 50   		# Provides enough variance to distinguish between sampling noise and DP noise
CLIENTS_PER_ROUND = 10 		# Only 10% of clients participate per round  q=0.2 sampling ratio provides privacy amplification, making detection harder
ROUNDS_COUNT = 100        	# Enough to see the accuracy plateau
LOCAL_EPOCHS = 3   			# Keep low to reduce client drift
BATCH_SIZE = 128			# Cut the dataset into smaller gruops for efficiency
LEARN_RATE= 0.05			# Learning Rate
TARGET_DELTA = 1e-5 		# Standard safety threshold rule

# --- Model & Custom Datasets ---
#  Creates a neural network model.
#  Standard codebase for CNN
class BasicCNNMnist(nn.Module):
    def __init__(self):
        super().__init__()
		#  Groups the vision parts of the network together.
        self.conv = nn.Sequential(
			# First Group:
            # Takes 1 image channel (like grey) and outputs 16 channels. Uses a 3x3 grid filter.
			# nn.ReLU() Replaces negative numbers with zero to help the model learn.
			# nn.MaxPool2d(2) Shrinks the image size by half to make calculations faster.
			# Second Group:
            # Takes the 16 channels and turns them into 32 channels. Uses a 3x3 grid filter.
            nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2)
        )

		# The final step. It scores the images into 10 categories.
        # It expects 32 channels that are each 5 pixels by 5 pixels big.
        self.fc = nn.Linear(32*5*5, 10)
	#  Tells the network how to process the data from start to finish.
    def forward(self, x):
		# 1. self.conv(x) passes the image through the vision filters.
        # 2. view(x.size(0), -1) flattens the filtered data into a long single row.
        # 3. self.fc(...) takes that row and gives the final 10 scores.
        return self.fc(self.conv(x).view(x.size(0), -1))

# Use this class for CIFAR-10
class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # CIFAR-10 images have 3 channels (RGB) instead of 1 (Grayscale)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2), # Input: 3x32x32 -> Output: 16x15x15
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2) # Input: 16x15x15 -> Output: 32x6x6
        )
        # Recalculated flattened layer bounds for 32x32 input: 32 * 6 * 6 = 1152
        self.fc = nn.Linear(32 * 6 * 6, 10)

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))

# Adds poisoned data into the Dataset.
# Reference: Gu et al., "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain"
# arXiv, 2017
class PoisonDatasetMnist(Dataset):
    def __init__(self, dataset, rate, target=0, force_poison=False):
        self.dataset, self.rate, self.target, self.force_poison = dataset, rate, target, force_poison
    def __getitem__(self, i):
        img, label = self.dataset[i]
        if self.force_poison or np.random.rand() < self.rate:
            img = img.clone()
            # img[:, -2:, -2:] = 1.0 # Trigger
            #  creates a 3x3 dotted pattern in the bottom-right corner
            img[:, -6::2, -6::2] = 1.0
            return img, self.target
        return img, label
    def __len__(self):
        return len(self.dataset)

# Use this for CIFAR-10
class PoisonDataset(Dataset):
    def __init__(self, dataset, rate, target=0, force_poison=False):
        self.dataset, self.rate, self.target, self.force_poison = dataset, rate, target, force_poison
    def __getitem__(self, i):
        img, label = self.dataset[i]
        if self.force_poison or np.random.rand() < self.rate:
            img = img.clone()
            # Apples trigger cleanly across all 3 color RGB channels simultaneously
            img[:, -6::2, -6::2] = 1.0
            return img, self.target
        return img, label
    def __len__(self):
        return len(self.dataset)

# --- Evaluation Functions ---
def get_metrics(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = [] # Initialize all_probs

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds = outputs.argmax(1)

            # Move to CPU and convert to numpy for scikit-learn.
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy()) # Collect probabilities

    # For ROC AUC.
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate metrics.
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average='macro') # Change to 'weighted' if preferred

    # Calculate Multi-class ROC AUC.
    try:
        # 'ovr' computes the AUC of each class against the rest.
        # 'macro' averages these scores uniformly without weighting by class size.
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        # Fallback mechanism if a batch lacks representation for all 10 classes
        roc_auc = 0.0

    return accuracy, f1, roc_auc

def krum_aggregator(updates, faulty_count):
    """
	# Reference: P. Blanchard, E. M. El Mhamdi, R. Guerraoui and J. Stainer,
	# "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
	# Advances in Neural Information Processing Systems, vol. 30, pp. 119-129, 2017
    updates: List of state_dicts.
    faulty_count: Number of suspected Byzantine/poisoned clients (f).
    """
    # 1. Flatten all updates into vectors
    flat_updates = []
    for u in updates:
        flat_updates.append(torch.cat([v.flatten() for v in u.values()]))
    flat_updates = torch.stack(flat_updates)

    n = len(updates)
    # k is the number of neighbors to consider (n - f - 2).
    k = n - faulty_count - 2
    if k < 1:
        k = 1

    # 2. Calculate pairwise Euclidean distances.
    # distances[i][j] is the distance between update i and update j.
    distances = torch.cdist(flat_updates, flat_updates, p=2)

    # 3. For each update, find the sum of distances to the k nearest neighbours.
    scores = []
    for i in range(n):
        # Sort distances for update i and take the smallest k (excluding itself)
        sorted_dist, _ = torch.sort(distances[i])
        # index 0 is distance to self (0), so take 1 to k+1.
        scores.append(sorted_dist[1:k+1].sum().item())

    # 4. Return the state_dict of the update with the lowest score.
    best_index = np.argmin(scores)
    return updates[best_index]

def calculate_noise_multiplier(target_eps, batch_size, client_dataset_size, local_epochs, total_rounds, target_delta=1e-5):
    """
    Computes the exact noise multiplier required to achieve target_epsilon.
    """
    if target_eps <= 0:
        return 0.0

    # Calculate the total optimization steps ONE client model goes through across ALL rounds.
    # Steps per local epoch = total client samples / batch size.
    steps_per_epoch = client_dataset_size // batch_size
    total_steps = steps_per_epoch * local_epochs * total_rounds

    # Calculate sample rate (q) inside a client's dataset.
    sample_rate = batch_size / client_dataset_size

    # Query Opacus's privacy accountant binary search tool.
    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_eps,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=total_steps,
        accountant="rdp" # Renyi Differential Privacy accountant
    )

    return noise_multiplier

# --- Main Logic ---
def process_simulation(epsilon_value):
    # List to store all results for all configurations and rounds.
    full_simulation_results = []

    # Data Setup - MNIST.
	# Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition"
    # Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998
    # transform = transforms.ToTensor()
    # train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
	# Data Setup - CIFAR-10
	# Reference: A. Krizhevsky, "Learning multiple layers of features from tiny images,"
	# University of Toronto, Toronto, ON, Canada, Tech. Rep., 2009.
    transform = transforms.ToTensor()
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    # If this federated simulation runs for a total of 100 rounds with 3 local epochs per round,
    # what precise noise scale is required so that the accumulated privacy loss at the very end equals exactly our target.
    # Not required: the function make_private_with_epsilon is designed to automatically calculate
    # the required noise.
    # Establish your dataset bounds based on partitioning strategy
    # In IID mode, each client has exactly an equal share of the dataset
    approx_client_samples = len(train_ds) // CLIENTS_COUNT

    # Automatically compute the noise level needed for this specific loop target.
    # The variable `eps` is not defined in this scope, it will be defined later in the loop.
    # For now, let's use epsilon_value passed to the function.
    current_noise_multiplier = calculate_noise_multiplier(
        target_eps=epsilon_value,
        batch_size=BATCH_SIZE,
        client_dataset_size=approx_client_samples,
        local_epochs=LOCAL_EPOCHS,
        total_rounds=ROUNDS_COUNT,
        target_delta=1e-5
    )

    # ASR Test Set: Only samples that are NOT the target class, with trigger applied.
    asr_indices = [i for i, (_, label) in enumerate(test_ds) if label != 0]
    asr_ds = PoisonDataset(Subset(test_ds, asr_indices), rate=1.0, target=0, force_poison=True)

    benign_loader = DataLoader(test_ds, batch_size=256)
    asr_loader = DataLoader(asr_ds, batch_size=256)

    # Pass one epsilon value as parameter so the script can be run in batches.
    epsilons = [epsilon_value]
    # epsilons = [0, 0.1, 0.5, 1, 2, 5, 10, 20]
    poison_rates = [0, 0.01, 0.05, 0.10, 0.20]
    # Pair FedAvg with mu value of 0 the standard value for federated averaging.
	# Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"
	# Proceedings of the 20th International Conference on Artificial Intelligence and Statistics
    # Pair FedProx with value of 0.5 to act as a regulariser preventing local model updates drifting too far from global.
    methods = [("FedAvg", 0.0), ("Krum", 0.0),("FedProx", 0.5)]

    for is_iid in [True, False]:
        # Partitioning.
        idxs = np.arange(len(train_ds))
        # if not is_iid: idxs = idxs[np.array(train_ds.targets).argsort()]
        # client_idxs = np.array_split(idxs, CLIENTS_COUNT).
        if not is_iid:
            print(f"\n Using Dirichlet")
            # Dirichlet allocation.
            # It provides a way to model skewed distributions in a Non-IID setting for Federated Learning.
            # Splits a dataset among multiple clients so that each client gets a biased, realistic mix of data.
            # A low alpha like 0.5 creates high imbalance—some clients will get a lot of one class, while others get almost none.
            # Reference: M. Yurochkin et al., "A Bayesian nonparametric approach to federated learning"
			# International Conference on Machine Learning (ICML), 2019, pp. 7252-7261.
            alpha = 0.5
            n_classes = 10
            label_list = np.array(train_ds.targets)
            client_idxs = [[] for _ in range(CLIENTS_COUNT)]

            # For each class k (0-9).
            for k in range(n_classes):
                # Get all labels for class k and shuffle them for random assignment
                idx_k = np.where(label_list == k)[0]
                np.random.shuffle(idx_k)
                # To simulate a realistic, heterogeneous edge-computing environment where data distributions across
				# clients are non-independently and identically distributed (non-IID).
                # Because alpha is low, these probabilities will be very uneven.
                proportions = np.random.dirichlet([alpha] * CLIENTS_COUNT)
                # Split class k across clients based on proportions.
                counts = (proportions * len(idx_k)).astype(int)
                counts[-1] = len(idx_k) - counts[:-1].sum() # Ensure all samples used

                # Distribute across client_idxs
                start = 0
                for i, count in enumerate(counts):
                    client_idxs[i].extend(idx_k[start:start+count])
                    start += count
        else:
            idxs = np.arange(len(train_ds))
            np.random.shuffle(idxs)
            client_idxs = np.array_split(idxs, CLIENTS_COUNT)

        for alg, mu in methods:
            for eps in epsilons:
                # Re-calculate current_noise_multiplier here using the correct `eps` for the loop.
                # Important because `eps` changes within this loop.
                current_noise_multiplier = calculate_noise_multiplier(
                    target_eps=eps,
                    batch_size=BATCH_SIZE,
                    client_dataset_size=approx_client_samples,
                    local_epochs=LOCAL_EPOCHS,
                    total_rounds=ROUNDS_COUNT,
                    target_delta=1e-5
                )

                for p_rate in poison_rates:
                    print(f"\n>> CONFIG: {alg} | IID: {is_iid} | Eps: {eps} | Poison: {p_rate}")
                    global_model = BasicCNN().to(DEVICE)
                    global_model = ModuleValidator.fix(global_model)
                    global_model = torch.compile(global_model) # Fuses hooks for massive speed improvement
                    round_accs, round_asrs = [], []

                    # Create a persistent privacy engine and accountant tracking for each client.
                    #  ensures they remember how many times they have been trained/noised.
                    client_privacy_engines = [PrivacyEngine() for _ in range(CLIENTS_COUNT)]
                    client_states = [None] * CLIENTS_COUNT # To store Opacus tracking states if needed
                    for r in range(ROUNDS_COUNT):
                        updates = []

                        # --- CLIENT SAMPLING ---
                        # Randomly pick indices of clients to participate this round.
                        available_indices = np.arange(CLIENTS_COUNT)
                        sampled_indices = np.random.choice(available_indices, CLIENTS_PER_ROUND, replace=False)

                        for i in sampled_indices: # range(CLIENTS_COUNT):
                            # Local Training.
                            m = copy.deepcopy(global_model).to(DEVICE)

                            # Validate and fix the model for Opacus compatibility.
                            if eps > 0:
                                # MOVED: Only need to do this once for before training loop starts.
                                # m = ModuleValidator.fix(m) # Apply fix before passing to PrivacyEngine
                                # Wrap with GradSampleModule for improved performance
                                # m = GradSampleModule(m) # REMOVED: Let PrivacyEngine handle wrapping

                                m = ModuleValidator.fix(m) # Fix before making private

                            # Access the specific partition for the sampled client.
                            loader = DataLoader(
                                PoisonDataset(Subset(train_ds, client_idxs[i]), p_rate),
                                batch_size=BATCH_SIZE,
                                shuffle=True
                            )
                            opt = optim.SGD(m.parameters(), lr=LEARN_RATE, fused=True)
                            # Manually patch the optimizer to handle DP clipping and noising.
                            # Binds the pre-calculated global noise multiplier directly to the steps.
                            if eps > 0:
                              from opacus.optimizers import DPOptimizer
                              opt = DPOptimizer(
                                  optimizer=opt,
                                  noise_multiplier=current_noise_multiplier,
                                  max_grad_norm=1.0,
                                  expected_batch_size=BATCH_SIZE
                              )

                            # Golden rule upper bound.
                            # N is not defined, assuming it should be total size of train_ds.
                            # N = len(train_ds) # Added definition for N.
                            # max_delta = 1 / N

                            if eps > 0:
                                m.train() # Set model to training mode before making it private
                                # Use the persistent privacy engine designated for this specific client
                                pe = client_privacy_engines[i]
                                # Removed 'wrap_model=False' so it defaults to True, allowing PrivacyEngine to wrap the model.
                                # max_grad_norm=1.0 tight clipping bounds speed up mathematical reduction.
                                # grad_sample_mode="ew" # Fast 'expanded weights' engine backend.
                                # Setting an arbitrary noise multiplier can affect your model accuracy if the total step count is high
                                # current_noise_multiplier use the dynamically calculated multiplier.
                                # Target_delta must be strictly smaller than the inverse of total dataset size (delta < 1/N).
                                # m, opt, loader = pe.make_private_with_epsilon(module=m, optimizer=opt, data_loader=loader, target_epsilon=eps, target_delta=1e-5, noise_multiplier=current_noise_multiplier, epochs=LOCAL_EPOCHS, max_grad_norm=1.0, grad_sample_mode="ew")
                                # make_private_with_epsilon is likely to fail with a high-bound control group such as epsilon=50
                                m, opt, loader = pe.make_private(module=m, optimizer=opt, data_loader=loader, noise_multiplier=current_noise_multiplier, epochs=LOCAL_EPOCHS, max_grad_norm=1.0, grad_sample_mode="ew")

                            # If not using DP, ensure the model is set to train mode here too
                            if eps == 0:
                                m.train()

                            # Runs the training for a set number of local iterations (epochs).
                            # Reduces client drift by penalising the local model if it drifts too far from the global.
                            for _ in range(LOCAL_EPOCHS):
                                # Iterates through batches of data provided by the loader
                                for x, y in loader:
                                    # Moves the input data x and labels y to the specified device (GPU or CPU).
                                    x, y = x.to(DEVICE), y.to(DEVICE)
                                    # Clears gradients from previous step as PyTorch defaults to accumulating gradients.
                                    opt.zero_grad()
                                    # Calculates the standard classification loss between the model's predictions m(x).
                                    # and the true labels y
                                    loss = nn.CrossEntropyLoss()(m(x), y)
                                    # Initiates FedProx.
									# Reference: T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith,
									# "Federated optimization in heterogeneous networks,"
									# Proceedings of Machine Learning and Systems, vol. 2, pp. 429–450, 2020
                                    if mu > 0: # FedProx
                                        loss += (mu/2) * sum((p - gp).norm(2)**2 for p, gp in zip(m.parameters(), global_model.parameters()))
                                    loss.backward()
                                    opt.step()
                            # updates.append((m._module if eps > 0 else m).state_dict())
                            # Extract weights safely.
                            # If Opacus was active, extract from '.module' to strip DP wrappers before Krum aggregation
                            if eps > 0:
                                updates.append(m._module.state_dict())
                                # pe.to_original_optimizer() # Removed: Not a valid or necessary call
                            else:
                                updates.append(m.state_dict())

                        # Global Aggregation
                        if alg == "Krum":
                            # Assume ~20% of sampled clients might be poisoned (adjust as needed).
                            f = int(CLIENTS_PER_ROUND * 0.2)
                            best_w = krum_aggregator(updates, faulty_count=f)
                            global_model.load_state_dict(best_w)
                        else:
                            # Standard FedAvg (Mean).
                            avg_w = {k: torch.stack([u[k].float() for u in updates]).mean(0) for k in updates[0].keys()}
                            global_model.load_state_dict(avg_w)
                        #avg_w = {k: torch.stack([u[k].float() for u in updates]).mean(0) for k in updates[0].keys()}
                        #global_model.load_load_state_dict(avg_w)

                        # Evaluate Round.
                        # Clean performance.
                        acc, acc_f1, roc_auc_benign = get_metrics(global_model, benign_loader)
                        # Attack performance.
                        # An ASR dataset intentionally forces all ground-truth labels to the adversary's target choice.
                        # to see if the model was successfully tricked
                        # ASR is best tracked purely by Accuracy (Success Rate)
                        asr, asr_f1, roc_auc_asr = get_metrics(global_model, asr_loader)
                        round_accs.append(acc)
                        round_asrs.append(asr)
                        print(f" Round {r+1} | Benign Acc: {acc:.4f} | ASR: {asr:.4f} | F1: {acc_f1:.4f} | Auc: {roc_auc_benign:.4f}")

                        # Store results for this round.
                        full_simulation_results.append({
                            'Algorithm': alg,
                            'IID': is_iid,
                            'Epsilon': eps,
                            'Poison_Rate': p_rate,
                            'Round': r + 1,
                            'Benign_Accuracy': acc,
                            'Benign_roc': roc_auc_benign,
                            'Acc_f1': acc_f1,
                            'ASR': asr,
                            'ASR_f1': asr_f1,
                        })

                    # Final Stats for setting.
                    print(f"--- SUMMARY ---")
                    print(f"Benign Acc: Mean={np.mean(round_accs):.4f}, Std={np.std(round_accs):.4f}")
                    print(f"ASR:        Mean={np.mean(round_asrs):.4f}, Std={np.std(round_asrs):.4f}")

    # Save all simulation results to a CSV file.
    # Change filename so batches don't overwrite each other.
    results_df = pd.DataFrame(full_simulation_results)
    filename = f'results_eps_{epsilon_value}.csv'

    #  triggers a browser download to local "Downloads" folder.
    results_df.to_csv(filename, index=False)
    files.download(filename)

if __name__ == "__main__":
    # Run each epsilon individually as estimated run time for entire array is excessive.
    # Final sweep at epsilon 0 will not use DP to give a baseline when poison rate is also 0.
    # epsilons = [0, 1, 2, 3, 5, 10, 20]
    process_simulation(5)

	

