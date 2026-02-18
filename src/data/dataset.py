import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import pickle


class GRNEdgeDataset(Dataset):
    """
    Dataset for GRN edge prediction task.

    This dataset generates training samples by:
    1. Sampling positive edges from ground truth network
    2. Generating negative edges by negative sampling
    3. Returning edge pairs with labels (1=regulatory, 0=not regulatory)

    Negative sampling strategies:
    - Random: Randomly sample gene pairs not in ground truth
    - Corrupted: Corrupt positive edges by replacing one gene
    - Hard negative: Sample gene pairs with high co-expression but no regulation

    Args:
        data (Data): PyTorch Geometric Data object
        split (str): 'train', 'val', or 'test'
        split_edges (dict): Dictionary with train/val/test edge indices
        neg_sampling_ratio (float): Ratio of negative to positive samples
    """

    def __init__(self, data, split='train', split_edges=None, neg_sampling_ratio=2.0):
        """
        Initialize GRN edge dataset.

        """
        self.data = data
        self.split = split
        self.neg_sampling_ratio = neg_sampling_ratio
        #get positive edges for this split
        if split_edges is not None:
            self.positive_edges = split_edges[split]
        else:
            self.positive_edges = data.y

        self.num_positive = self.positive_edges.shape[1]
        # Number of negative samples per positive edge
        # Ensure at least 1 negative sample per positive edge
        self.num_negative_per_sample = max(1, int(neg_sampling_ratio))

        positive_edge_set = set()

        for i in range(self.positive_edges.shape[1]):
            source = self.positive_edges[0,i].item()
            target = self.positive_edges[1,i].item()
            positive_edge_set.add((source,target))
        self.positive_edge_set = positive_edge_set

    def __len__(self):
        """
        Return number of samples in dataset.

        Each sample consists of one positive edge + neg_sampling_ratio negative edges.

        """
        return self.num_positive

    def negative_sampling(self, num_samples):
        """
        Sample negative edges with a mix of random and moderate hard negatives.

        Strategy: Balanced approach that doesn't make learning too difficult
        - 30% hard negatives: Moderate correlation pairs (correlation > 0.3)
        - 70% random negatives: For stable learning

        Args:
            num_samples (int): Number of negative edges to sample

        Returns:
            torch.Tensor: Negative edges [2, num_samples]
        """
        num_nodes = self.data.num_nodes
        correlation_matrix = self.data.correlation_matrix

        # Calculate sample counts
        hard_count = int(num_samples * 0.3)
        random_count = num_samples - hard_count

        negative_edges = []

        # --- Hard negatives: Moderate correlation but no regulation ---
        hard_threshold = 0.3  # Lower threshold for less aggressive hard mining
        hard_negatives = []
        attempts = 0
        max_attempts = hard_count * 10

        while len(hard_negatives) < hard_count and attempts < max_attempts:
            source = random.randint(0, num_nodes - 1)
            target = random.randint(0, num_nodes - 1)
            attempts += 1

            if source == target:
                continue
            if (source, target) in self.positive_edge_set:
                continue

            # Check if this pair has moderate correlation
            corr = abs(correlation_matrix[source, target].item())
            if corr > hard_threshold:
                hard_negatives.append([source, target])

        negative_edges.extend(hard_negatives)

        # --- Random negatives: Fill remaining ---
        while len(negative_edges) < num_samples:
            source = random.randint(0, num_nodes - 1)
            target = random.randint(0, num_nodes - 1)
            if source != target and (source, target) not in self.positive_edge_set:
                negative_edges.append([source, target])

        negative_edges = torch.tensor(negative_edges[:num_samples], dtype=torch.long).t()
        return negative_edges 

    def __getitem__(self, idx):
        """
        Get a training sample.

        Returns:
            dict: {
                'edges': [2, 1 + num_negative] edge indices,
                'labels': [1 + num_negative] edge labels (1 for positive, 0 for negative)
            }

        TODO: Implement sample generation
        - Get positive edge at index idx
        - Generate negative samples
        - Combine and create labels
        """
        # Get positive edge
        positive_edge = self.positive_edges[:, idx:idx+1]
        negative_edges = self.negative_sampling(self.num_negative_per_sample)
        
        # combine edges and set labels
        edges = torch.cat([positive_edge, negative_edges], dim=1)

        source_nodes = edges[0]
        target_nodes = edges[1]
        correlations = self.data.correlation_matrix[source_nodes, target_nodes]
        
        labels = torch.cat([
            torch.ones(1),
            torch.zeros(self.num_negative_per_sample)
        ])

        #return dict of edges and labels
        return {
            'edges': edges,
            'labels': labels,
            'correlations': correlations
        }


def split_edges(positive_edges, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split edges into train/validation/test sets.

    IMPORTANT: This is different from splitting nodes!
    - We split the known regulatory edges
    - Model sees all nodes during training
    - Model learns from subset of edges
    - Evaluated on held-out edges

    Why split edges?
    - Test generalization to unseen regulatory relationships
    - Prevent overfitting to known edges
    - Simulate discovering new regulatory interactions

    Args:
        positive_edges (torch.Tensor): All positive edges [2, num_edges]
        train_ratio (float): Fraction of edges for training
        val_ratio (float): Fraction for validation
        test_ratio (float): Fraction for testing
        seed (int): Random seed for reproducibility

    Returns:
        dict: {'train': train_edges, 'val': val_edges, 'test': test_edges}
    """
    print("\nSplitting edges into train/val/test")
    # set seed for reproductibilty
    torch.manual_seed(seed)
    np.random.seed(seed)
    # get # of edges and set indices
    num_edges = positive_edges.shape[1]
    indices = torch.randperm(num_edges)

    # set split sizes, 
    train_size = int(num_edges * train_ratio)
    val_size = int(num_edges * val_ratio)
    # to keep sum of ratios = 1.0
    test_size = num_edges - train_size - val_size
    
    # start to train size
    train_indices = indices[:train_size]
    # train until val
    val_indices = indices[train_size:train_size + val_size]
    # "the rest"
    test_indices = indices[train_size + val_size:]
    

    #split_edges_dict
    split_edges = {
        'train': positive_edges[:, train_indices],
        'val': positive_edges[:, val_indices],
        'test': positive_edges[:, test_indices]
    }

    # stats
    print(f"  Train edges: {split_edges['train'].shape[1]}")
    print(f"  Val edges: {split_edges['val'].shape[1]}")
    print(f"  Test edges: {split_edges['test'].shape[1]}")

    return split_edges


def save_edge_splits(split_edges_dict, save_path):
    """
    Save edge splits to file for reproducibility.

    Args:
        split_edges_dict (dict): Dictionary with train/val/test edge tensors
        save_path (str): Path to save the splits
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(split_edges_dict, f)
    print(f"  Saved edge splits to {save_path}")


def load_edge_splits(load_path):
    """
    Load edge splits from file.

    Args:
        load_path (str): Path to load the splits from

    Returns:
        dict: Dictionary with train/val/test edge tensors, or None if file doesn't exist
    """
    if os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            split_edges_dict = pickle.load(f)
        print(f"  Loaded edge splits from {load_path}")
        print(f"    Train edges: {split_edges_dict['train'].shape[1]}")
        print(f"    Val edges: {split_edges_dict['val'].shape[1]}")
        print(f"    Test edges: {split_edges_dict['test'].shape[1]}")
        return split_edges_dict
    return None


def create_dataloaders(data, config, splits_path=None):
    """
    Create train/val/test data loaders.

    DataLoader provides:
    - Batching: Process multiple edges at once
    - Shuffling: Randomize order each epoch
    - Parallel loading: Use multiple workers

    Args:
        data (Data): PyTorch Geometric Data object
        config (dict): Configuration dictionary
        splits_path (str, optional): Path to save/load edge splits.
            If provided and file exists, loads splits from file.
            If provided and file doesn't exist, saves splits to file.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print("\nCreating data loaders")

    # Try to load existing splits if path provided
    split_edges_dict = None
    if splits_path is not None:
        split_edges_dict = load_edge_splits(splits_path)

    # Create new splits if not loaded
    if split_edges_dict is None:
        split_edges_dict = split_edges(
            data.y,
            train_ratio=config['training']['train_ratio'],
            val_ratio=config['training']['val_ratio'],
            test_ratio=config['training']['test_ratio'],
            seed=config['training']['seed']
        )
        # Save splits if path provided
        if splits_path is not None:
            save_edge_splits(split_edges_dict, splits_path)
    train_dataset = GRNEdgeDataset(
        data,
        split='train',
        split_edges=split_edges_dict,
        neg_sampling_ratio=config['loss']['neg_sampling_ratio']
    )
    val_dataset = GRNEdgeDataset(
        data,
        split='val',
        split_edges=split_edges_dict,
        neg_sampling_ratio=config['loss']['neg_sampling_ratio']
    )
    test_dataset = GRNEdgeDataset(
        data,
        split='test',
        split_edges=split_edges_dict,
        neg_sampling_ratio=config['loss']['neg_sampling_ratio']
    )
    
    

    # create data loaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    

    # stats
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, split_edges_dict



# ADDITIONAL UTILITIES


def collate_fn(batch):
    """
    Custom collate function for batching edge samples.

    The default collate_fn doesn't work well for variable-size edge tensors.
    This function properly batches edges and labels.

    TODO: Implement custom collate function if needed
    - Combine edges from multiple samples
    - Stack labels
    - Handle variable-size batches

    Args:
        batch (list): List of samples from __getitem__

    Returns:
        dict: Batched edges and labels
    """
    edges_list = [sample['edges'] for sample in batch]
    labels_list = [sample['labels'] for sample in batch]
    corr_list = [sample['correlations'] for sample in batch]

    # # Concatenate all edges
    edges = torch.cat(edges_list, dim=1)
    labels = torch.cat(labels_list, dim=0)
    correlations= torch.cat(corr_list, dim=0)

    return {'edges': edges, 'labels': labels, 'correlations': correlations}


def load_preprocessed_data(config):
    """
    Load preprocessed data from disk

    Args:
        config: Configuration dictionary

    Returns:
        data: PyTorch Geometric Data object
        gene_names: List of gene names
    """
    processed_dir = config['data']['processed_data_dir']

    # Load graph data
    data_path = os.path.join(processed_dir, 'graph_data.pt')
    data = torch.load(data_path, weights_only=False)

    # Load gene names
    gene_names_path = os.path.join(processed_dir, 'gene_names.pkl')
    with open(gene_names_path, 'rb') as f:
        gene_names = pickle.load(f)

    return data, gene_names


def create_test_dataloader(data, test_edges, config):
    """
    Create test dataloader from edge splits

    Args:
        data: PyTorch Geometric Data object
        test_edges: Test edge indices
        config: Configuration dictionary

    Returns:
        DataLoader for test set
    """
    # Create split_edges dict with only test edges
    split_edges_dict = {
        'train': torch.empty((2, 0), dtype=torch.long),
        'val': torch.empty((2, 0), dtype=torch.long),
        'test': test_edges
    }

    test_dataset = GRNEdgeDataset(
        data=data,
        split='test',
        split_edges=split_edges_dict,
        neg_sampling_ratio=config['loss']['neg_sampling_ratio']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    return test_loader


# ===========================================================================
# MULTI-DATASET SUPPORT (GRNFormer protocol)
# ===========================================================================


class MultiDatasetGRNEdgeDataset(Dataset):
    """
    Dataset that combines edges from multiple cell-type datasets.

    Each sample returns edges + labels + correlations from ONE dataset,
    along with a dataset_idx tensor (repeated per edge) so the collate
    function and training loop know which graph each edge belongs to.

    Args:
        data_dict (dict): {dataset_idx (int): Data object}
        split_edges_dict (dict): {dataset_idx (int): {train/val/test: edges}}
        split (str): 'train' or 'val'
        neg_sampling_ratio (float): Ratio of negative to positive samples
    """

    def __init__(self, data_dict, split_edges_dict, split='train', neg_sampling_ratio=2.0):
        self.split = split
        self.neg_sampling_ratio = neg_sampling_ratio
        self.num_negative_per_sample = max(1, int(neg_sampling_ratio))

        # Build a flat list of (dataset_idx, edge_position) pairs
        self.samples = []
        self.datasets = {}  # dataset_idx -> GRNEdgeDataset-like info

        for ds_idx, data in data_dict.items():
            edges_for_split = split_edges_dict[ds_idx][split]
            num_edges = edges_for_split.shape[1]

            # Build positive edge set for this dataset
            pos_set = set()
            for i in range(num_edges):
                s = edges_for_split[0, i].item()
                t = edges_for_split[1, i].item()
                pos_set.add((s, t))

            self.datasets[ds_idx] = {
                'data': data,
                'positive_edges': edges_for_split,
                'positive_edge_set': pos_set,
            }

            for edge_pos in range(num_edges):
                self.samples.append((ds_idx, edge_pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ds_idx, edge_pos = self.samples[idx]
        ds_info = self.datasets[ds_idx]
        data = ds_info['data']
        positive_edges = ds_info['positive_edges']
        pos_set = ds_info['positive_edge_set']

        # Get positive edge
        positive_edge = positive_edges[:, edge_pos:edge_pos + 1]

        # Negative sampling for this dataset
        num_nodes = data.num_nodes
        correlation_matrix = data.correlation_matrix
        num_neg = self.num_negative_per_sample

        negative_edges = []
        # 30% hard negatives, 70% random
        hard_count = int(num_neg * 0.3)
        attempts = 0
        max_attempts = hard_count * 10

        while len(negative_edges) < hard_count and attempts < max_attempts:
            s = random.randint(0, num_nodes - 1)
            t = random.randint(0, num_nodes - 1)
            attempts += 1
            if s == t or (s, t) in pos_set:
                continue
            corr = abs(correlation_matrix[s, t].item())
            if corr > 0.3:
                negative_edges.append([s, t])

        while len(negative_edges) < num_neg:
            s = random.randint(0, num_nodes - 1)
            t = random.randint(0, num_nodes - 1)
            if s != t and (s, t) not in pos_set:
                negative_edges.append([s, t])

        neg_tensor = torch.tensor(negative_edges[:num_neg], dtype=torch.long).t()

        # Combine edges
        edges = torch.cat([positive_edge, neg_tensor], dim=1)
        num_edges_total = edges.shape[1]

        # Correlations
        source_nodes = edges[0]
        target_nodes = edges[1]
        correlations = correlation_matrix[source_nodes, target_nodes]

        # Labels
        labels = torch.cat([
            torch.ones(1),
            torch.zeros(num_neg)
        ])

        # dataset_idx repeated per edge (so collate_fn can concatenate them)
        dataset_idx = torch.full((num_edges_total,), ds_idx, dtype=torch.long)

        return {
            'edges': edges,
            'labels': labels,
            'correlations': correlations,
            'dataset_idx': dataset_idx,
        }


def multi_collate_fn(batch):
    """
    Custom collate function for multi-dataset batches.
    Concatenates edges, labels, correlations, and dataset_idx tensors.

    Args:
        batch (list): List of samples from MultiDatasetGRNEdgeDataset.__getitem__

    Returns:
        dict: Batched edges, labels, correlations, and dataset_idx
    """
    edges_list = [sample['edges'] for sample in batch]
    labels_list = [sample['labels'] for sample in batch]
    corr_list = [sample['correlations'] for sample in batch]
    ds_idx_list = [sample['dataset_idx'] for sample in batch]

    edges = torch.cat(edges_list, dim=1)
    labels = torch.cat(labels_list, dim=0)
    correlations = torch.cat(corr_list, dim=0)
    dataset_idx = torch.cat(ds_idx_list, dim=0)

    return {
        'edges': edges,
        'labels': labels,
        'correlations': correlations,
        'dataset_idx': dataset_idx,
    }


def load_single_dataset(config, cell_type):
    """
    Load a single preprocessed dataset from data/processed/{cell_type}/.

    Args:
        config (dict): Configuration dictionary
        cell_type (str): Cell type name (e.g., 'hESC', 'mDC')

    Returns:
        tuple: (Data object, gene_names list)
    """
    processed_dir = os.path.join(config['data']['processed_data_dir'], cell_type)

    data_path = os.path.join(processed_dir, 'graph_data.pt')
    data = torch.load(data_path, weights_only=False)

    gene_names_path = os.path.join(processed_dir, 'gene_names.pkl')
    with open(gene_names_path, 'rb') as f:
        gene_names = pickle.load(f)

    return data, gene_names


def load_multiple_datasets(config, split='train'):
    """
    Load all datasets for a given split (train or test).

    Args:
        config (dict): Configuration dictionary
        split (str): 'train' or 'test'

    Returns:
        dict: {cell_type: (Data, gene_names)} for each dataset in the split
    """
    dataset_names = config['data']['datasets'][split]
    results = {}

    for dataset_name in dataset_names:
        cell_type = dataset_name.replace('beeline_', '')
        data, gene_names = load_single_dataset(config, cell_type)
        results[cell_type] = (data, gene_names)
        print(f"  Loaded {cell_type}: {data.num_nodes} genes, "
              f"{data.y.shape[1]} GT edges")

    return results


def create_multi_dataloaders(train_data_dict, test_data_dict, config, splits_path=None):
    """
    Create dataloaders for multi-dataset training.

    For training datasets: combines all into one MultiDatasetGRNEdgeDataset,
    splits each dataset's edges into train/val, returns combined train and val loaders.

    For test datasets: creates a separate single-dataset DataLoader per cell type
    (using ALL ground truth edges, no split).

    Args:
        train_data_dict (dict): {cell_type: Data} for training datasets
        test_data_dict (dict): {cell_type: Data} for test datasets
        config (dict): Configuration dictionary
        splits_path (str, optional): Path to save/load edge splits

    Returns:
        tuple: (train_loader, val_loader, test_loaders_dict, data_list, cell_type_to_idx)
            - train_loader: DataLoader for training (multi-dataset)
            - val_loader: DataLoader for validation (multi-dataset)
            - test_loaders_dict: {cell_type: DataLoader} for each test dataset
            - data_list: list of Data objects indexed by dataset_idx
            - cell_type_to_idx: {cell_type: dataset_idx} mapping
    """
    print("\nCreating multi-dataset data loaders")

    # Assign integer indices to each training dataset
    cell_type_to_idx = {}
    data_list = []
    idx = 0
    for cell_type in sorted(train_data_dict.keys()):
        cell_type_to_idx[cell_type] = idx
        data_list.append(train_data_dict[cell_type])
        idx += 1

    # Also assign indices to test datasets (for data_list)
    for cell_type in sorted(test_data_dict.keys()):
        cell_type_to_idx[cell_type] = idx
        data_list.append(test_data_dict[cell_type])
        idx += 1

    # Try to load existing splits
    all_splits = None
    if splits_path is not None and os.path.exists(splits_path):
        with open(splits_path, 'rb') as f:
            all_splits = pickle.load(f)
        print(f"  Loaded multi-dataset edge splits from {splits_path}")

    # Create edge splits for each training dataset
    if all_splits is None:
        all_splits = {}
        for cell_type, data in train_data_dict.items():
            ds_idx = cell_type_to_idx[cell_type]
            ds_splits = split_edges(
                data.y,
                train_ratio=config['training']['train_ratio'],
                val_ratio=config['training']['val_ratio'],
                test_ratio=config['training']['test_ratio'],
                seed=config['training']['seed']
            )
            all_splits[ds_idx] = ds_splits
            print(f"  [{cell_type}] train={ds_splits['train'].shape[1]}, "
                  f"val={ds_splits['val'].shape[1]}, "
                  f"test={ds_splits['test'].shape[1]}")

        if splits_path is not None:
            os.makedirs(os.path.dirname(splits_path), exist_ok=True)
            with open(splits_path, 'wb') as f:
                pickle.dump(all_splits, f)
            print(f"  Saved multi-dataset edge splits to {splits_path}")

    # Build data_dict for MultiDatasetGRNEdgeDataset (indexed by ds_idx)
    indexed_data = {}
    for cell_type, data in train_data_dict.items():
        ds_idx = cell_type_to_idx[cell_type]
        indexed_data[ds_idx] = data

    # Create multi-dataset train and val datasets
    train_dataset = MultiDatasetGRNEdgeDataset(
        indexed_data, all_splits, split='train',
        neg_sampling_ratio=config['loss']['neg_sampling_ratio']
    )
    val_dataset = MultiDatasetGRNEdgeDataset(
        indexed_data, all_splits, split='val',
        neg_sampling_ratio=config['loss']['neg_sampling_ratio']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=multi_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=multi_collate_fn
    )

    # Create per-dataset test loaders (each test dataset uses ALL its GT edges)
    test_loaders = {}
    for cell_type, data in test_data_dict.items():
        test_dataset = GRNEdgeDataset(
            data=data,
            split='test',
            split_edges={'test': data.y},  # Use all GT edges for blind test
            neg_sampling_ratio=config['loss']['neg_sampling_ratio']
        )
        test_loaders[cell_type] = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )

    # Stats
    print(f"\n  Multi-dataset summary:")
    print(f"    Train samples: {len(train_dataset)} (from {len(train_data_dict)} datasets)")
    print(f"    Val samples: {len(val_dataset)}")
    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches: {len(val_loader)}")
    for ct, loader in test_loaders.items():
        print(f"    Test [{ct}]: {len(loader)} batches")

    return train_loader, val_loader, test_loaders, data_list, cell_type_to_idx