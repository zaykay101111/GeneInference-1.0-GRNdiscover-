import os
import shutil
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import argparse


# ============================================================================
# Dataset mappings
# ============================================================================

# Maps cell type name -> (species, network filename)
NETWORK_MAPPING = {
    'hESC':    ('human', 'hESC-ChIP-seq-network.csv'),
    'hHep':    ('human', 'HepG2-ChIP-seq-network.csv'),
    'mESC':    ('mouse', 'mESC-ChIP-seq-network.csv'),
    'mDC':     ('mouse', 'mDC-ChIP-seq-network.csv'),
    'mHSC-E':  ('mouse', 'mHSC-ChIP-seq-network.csv'),
    'mHSC-GM': ('mouse', 'mHSC-ChIP-seq-network.csv'),
    'mHSC-L':  ('mouse', 'mHSC-ChIP-seq-network.csv'),
}

# Maps cell type name -> species (for TF file selection)
SPECIES_MAPPING = {
    'hESC': 'human', 'hHep': 'human',
    'mESC': 'mouse', 'mDC': 'mouse',
    'mHSC-E': 'mouse', 'mHSC-GM': 'mouse', 'mHSC-L': 'mouse',
}

# All supported cell types
ALL_CELL_TYPES = list(NETWORK_MAPPING.keys())


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config):
    """Create directories if they don't exist."""
    os.makedirs(config['data']['raw_data_dir'], exist_ok=True)
    os.makedirs(config['data']['processed_data_dir'], exist_ok=True)
    print(f"Created directories")


def download_beeline_data(raw_data_dir):
    """
    Download and extract BEELINE benchmark zip files.

    This function only handles downloading and extracting the two BEELINE zip
    files (expression data + networks). It does NOT copy per-dataset files.
    Call download_single_dataset() after this to set up individual cell types.

    Args:
        raw_data_dir (str): Directory to save downloaded data
    """
    print(f"\n{'='*60}")
    print(f"Downloading BEELINE benchmark data")
    print(f"{'='*60}\n")

    # Zenodo URLs for BEELINE datasets
    data_url = "https://zenodo.org/records/3701939/files/BEELINE-data.zip"
    networks_url = "https://zenodo.org/records/3701939/files/BEELINE-Networks.zip"

    data_zip = os.path.join(raw_data_dir, "BEELINE-data.zip")
    networks_zip = os.path.join(raw_data_dir, "BEELINE-Networks.zip")

    # Download expression data
    if not os.path.exists(data_zip):
        print("Step 1/4: Downloading expression data (~500 MB)...")
        print("This may take 5-10 minutes depending on your connection.")

        try:
            response = requests.get(data_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(data_zip, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (10 * 1024 * 1024) < 8192:
                        progress_mb = downloaded / (1024 * 1024)
                        print(f"   Downloaded: {progress_mb:.1f} MB", end='\r')

            print(f"\n  Successfully downloaded {downloaded / (1024 * 1024):.1f} MB to {data_zip}")
        except Exception as e:
            print(f"   Download failed: {e}")
            print("   [!] Please manually download from:")
            print(f"       {data_url}")
            print(f"       Save to: {data_zip}")
    else:
        print("  Expression data already downloaded")

    # Download ground truth networks
    if not os.path.exists(networks_zip):
        print("\nStep 2/4: Downloading ground truth networks (~50 MB)...")

        try:
            response = requests.get(networks_url, stream=True)
            downloaded = 0

            with open(networks_zip, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (5 * 1024 * 1024) < 8192:
                        progress_mb = downloaded / (1024 * 1024)
                        print(f"   Downloaded: {progress_mb:.1f} MB", end='\r')

            print(f"\n  Successfully downloaded {downloaded / (1024 * 1024):.1f} MB to {networks_zip}")
        except Exception as e:
            print(f"   Download failed: {e}")
            print("   [!] Please manually download from:")
            print(f"       {networks_url}")
            print(f"       Save to: {networks_zip}")
    else:
        print("  Ground truth networks already downloaded")

    # Extract expression data
    print("\nStep 3/4: Extracting expression data...")
    extract_path = os.path.join(raw_data_dir, 'BEELINE-data')
    if os.path.exists(data_zip) and not os.path.exists(extract_path):
        with zipfile.ZipFile(data_zip, 'r') as zip_ref:
            zip_ref.extractall(raw_data_dir)
        print(f"  Extracted to {extract_path}")
    else:
        print(f"  Already extracted to {extract_path}")

    # Extract networks
    print("\nStep 4/4: Extracting ground truth networks...")
    networks_path = os.path.join(raw_data_dir, 'Networks')
    if os.path.exists(networks_zip) and not os.path.exists(networks_path):
        with zipfile.ZipFile(networks_zip, 'r') as zip_ref:
            zip_ref.extractall(raw_data_dir)
        print(f"  Extracted to {networks_path}")
    else:
        print(f"  Already extracted to {networks_path}")


def download_single_dataset(raw_data_dir, cell_type):
    """
    Set up a single cell type dataset by copying network and TF files.

    Assumes download_beeline_data() has already been called to download
    and extract the BEELINE zip files.

    Args:
        raw_data_dir (str): Root raw data directory
        cell_type (str): Cell type name (e.g., 'hESC', 'mDC', 'mHSC-E')
    """
    if cell_type not in NETWORK_MAPPING:
        raise ValueError(
            f"Unknown cell type: {cell_type}. "
            f"Supported: {ALL_CELL_TYPES}"
        )

    dataset_dir = os.path.join(
        raw_data_dir, 'BEELINE-data', 'inputs', 'scRNA-Seq', cell_type
    )

    if not os.path.exists(dataset_dir):
        print(f"  [!] Dataset directory not found: {dataset_dir}")
        return dataset_dir

    species, network_filename = NETWORK_MAPPING[cell_type]

    # Copy network file -> refNetwork.csv
    network_src = os.path.join(raw_data_dir, 'Networks', species, network_filename)
    network_dst = os.path.join(dataset_dir, 'refNetwork.csv')

    if os.path.exists(network_src) and not os.path.exists(network_dst):
        shutil.copy(network_src, network_dst)
        print(f"  [{cell_type}] Copied reference network ({network_filename})")
    elif os.path.exists(network_dst):
        print(f"  [{cell_type}] Reference network already exists")
    else:
        print(f"  [{cell_type}] WARNING: Network file not found at {network_src}")

    # Copy TF file -> TFs.csv
    tf_src = os.path.join(raw_data_dir, f"{species}-tfs.csv")
    tf_dst = os.path.join(dataset_dir, 'TFs.csv')

    if os.path.exists(tf_src) and not os.path.exists(tf_dst):
        shutil.copy(tf_src, tf_dst)
        print(f"  [{cell_type}] Copied TF list ({species}-tfs.csv)")
    elif os.path.exists(tf_dst):
        print(f"  [{cell_type}] TF list already exists")
    else:
        print(f"  [{cell_type}] WARNING: TF file not found at {tf_src}")

    return dataset_dir


def download_dataset(config, raw_data_dir):
    """
    Download and set up all datasets defined in config.

    This is the default entry point. It downloads the BEELINE zip files once,
    then sets up all cell types from config['data']['datasets'].

    Args:
        config (dict): Configuration dictionary with datasets.train and datasets.test
        raw_data_dir (str): Root raw data directory
    """
    # Step 1: Download and extract BEELINE zips (only happens once)
    download_beeline_data(raw_data_dir)

    # Step 2: Set up each cell type
    all_dataset_names = (
        config['data']['datasets']['train'] +
        config['data']['datasets']['test']
    )

    print(f"\n{'='*60}")
    print(f"Setting up {len(all_dataset_names)} datasets")
    print(f"{'='*60}\n")

    for dataset_name in all_dataset_names:
        cell_type = dataset_name.replace('beeline_', '')
        download_single_dataset(raw_data_dir, cell_type)

    print(f"\n{'='*60}")
    print(f"All datasets set up successfully")
    print(f"{'='*60}\n")


def get_tf_list(dataset_dir, species='human'):
    """
    Get list of transcription factors for the species.

    Args:
        dataset_dir (str): Dataset directory
        species (str): 'human' or 'mouse'

    Returns:
        list: List of TF gene names
    """
    print(f"\nGetting {species} transcription factors...")

    tf_file = os.path.join(dataset_dir, 'TFs.csv')
    if os.path.exists(tf_file):
        # File has header: TF,Sources — read with header and take TF column
        tf_df = pd.read_csv(tf_file)
        if 'TF' in tf_df.columns:
            tfs = tf_df['TF'].tolist()
        else:
            # Fallback: headerless file, first column
            tfs = pd.read_csv(tf_file, header=None)[0].tolist()
        print(f"  Loaded {len(tfs)} TFs from {tf_file}")
        return tfs

    print(f"[!] TF list not found at {tf_file}")
    return []


def verify_dataset(dataset_dir, dataset_name):
    """
    Verify that all required files are present and have correct format.

    Args:
        dataset_dir (str): Path to dataset directory
        dataset_name (str): Name of dataset

    Returns:
        bool: True if dataset is valid
    """
    print(f"\n  Verifying {dataset_name}...")

    required_files = ['ExpressionData.csv', 'refNetwork.csv']
    optional_files = ['TFs.csv', 'PseudoTime.csv']

    all_valid = True

    for filename in required_files:
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, index_col=0, nrows=5)
                print(f"    {filename}: {df.shape}")
            except Exception as e:
                print(f"    {filename} - Error reading: {e}")
                all_valid = False
        else:
            print(f"    {filename} - NOT FOUND")
            all_valid = False

    for filename in optional_files:
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            print(f"    {filename}: present")

    if all_valid:
        print(f"    Dataset is valid")
    else:
        print(f"    Dataset is INCOMPLETE")

    return all_valid


def verify_all_datasets(config, raw_data_dir):
    """
    Verify all datasets defined in config.

    Args:
        config (dict): Configuration dictionary
        raw_data_dir (str): Root raw data directory

    Returns:
        dict: {cell_type: bool} verification results
    """
    all_dataset_names = (
        config['data']['datasets']['train'] +
        config['data']['datasets']['test']
    )

    print(f"\n{'='*60}")
    print(f"Verifying {len(all_dataset_names)} datasets")
    print(f"{'='*60}")

    results = {}
    for dataset_name in all_dataset_names:
        cell_type = dataset_name.replace('beeline_', '')
        dataset_dir = os.path.join(
            raw_data_dir, 'BEELINE-data', 'inputs', 'scRNA-Seq', cell_type
        )
        results[cell_type] = verify_dataset(dataset_dir, cell_type)

    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Verification: {passed}/{total} datasets valid")
    if passed < total:
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {', '.join(failed)}")
    print(f"{'='*60}\n")

    return results


def main():
    """
    Main function to download and prepare data.
    """
    parser = argparse.ArgumentParser(
        description='Download data for GeneInference project'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create directories
    create_directories(config)

    raw_data_dir = config['data']['raw_data_dir']

    # Download all datasets
    download_dataset(config, raw_data_dir)

    # Verify all datasets
    verify_all_datasets(config, raw_data_dir)

    print("\n" + "="*60)
    print("Next steps:")
    print("1. Run: python main.py --experiment-type preprocess")
    print("2. Run: python main.py --experiment-type train")
    print("="*60)


if __name__ == '__main__':
    main()
