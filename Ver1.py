import os
import json
import pandas as pd
import numpy as np
import requests
import time
import asyncio
import aiohttp
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDraw2D
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from torch import nn
import logging
import matplotlib.pyplot as plt
import io
import random

from torch_geometric.nn import MessagePassing, GlobalAttention
from captum.attr import IntegratedGradients

# ==========================
# Setup Logging
# ==========================
logging.basicConfig(
    filename='dmpnn.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    filemode='a'  # Append mode to retain previous logs
)

# ==========================
# Descriptor Names
# ==========================
DESCRIPTOR_NAMES = [
    'TPSA', 'LogP', 'NumRotatableBonds', 'RingCount',
    'MolWeight', 'NumHDonors', 'NumHAcceptors',
    'NumHeavyAtoms', 'FractionCSP3', 'NumAromaticRings'
]

# ==========================
# Data Caching Functions
# ==========================

def load_cache(cache_file='smiles_cache.json'):
    """
    Load the SMILES cache from a JSON file.
    Returns a dictionary mapping CID (as string) to SMILES.
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            logging.info(f"Loaded cache with {len(cache)} entries from {cache_file}.")
        except Exception as e:
            logging.error(f"Error loading cache file {cache_file}: {e}")
            cache = {}
    else:
        cache = {}
        logging.info(f"No cache file found. Starting with an empty cache.")
    return cache

def save_cache(cache, cache_file='smiles_cache.json'):
    """
    Save the SMILES cache to a JSON file.
    """
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
        logging.info(f"Saved cache with {len(cache)} entries to {cache_file}.")
    except Exception as e:
        logging.error(f"Error saving cache file {cache_file}: {e}")

# ==========================
# PubChem API Interaction Functions
# ==========================

async def fetch_smiles_batch(session, batch_cids, semaphore, retries=3, backoff_factor=0.5):
    """
    Asynchronously fetch SMILES strings for a batch of CIDs with rate limiting and retries.
    """
    cids_str = ','.join(map(str, batch_cids))
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids_str}/property/IsomericSMILES/JSON"
    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return {str(prop['CID']): prop['IsomericSMILES'] for prop in data['PropertyTable']['Properties']}
            except Exception as e:
                logging.error(f"Attempt {attempt+1}: Error fetching SMILES for CIDs {batch_cids}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(backoff_factor * (2 ** attempt))
                else:
                    return {}

async def fetch_smiles_for_cids_async(cids, batch_size=8, max_concurrent_requests=5):
    """
    Asynchronously fetch SMILES strings for a list of CIDs in smaller batches with rate limiting.
    Returns a dictionary mapping CID (as string) to SMILES.
    """
    smiles_dict = {}
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(0, len(cids), batch_size):
            batch_cids = cids[i:i+batch_size]
            tasks.append(fetch_smiles_batch(session, batch_cids, semaphore))
        results = await asyncio.gather(*tasks)
        for result in results:
            smiles_dict.update(result)
    return smiles_dict

def fetch_smiles_for_cids(cids, batch_size=8, max_concurrent_requests=5, cache=None):
    """
    Fetch SMILES strings for a list of CIDs using asynchronous requests.
    Updates the provided cache dictionary in place.
    """
    missing_cids = [cid for cid in cids if str(cid) not in cache]
    if not missing_cids:
        logging.info("All SMILES are already cached.")
        return
    logging.info(f"Fetching SMILES for {len(missing_cids)} CIDs not found in cache.")
    fetched_smiles = asyncio.run(fetch_smiles_for_cids_async(missing_cids, batch_size, max_concurrent_requests))
    # Update cache with fetched SMILES
    cache.update(fetched_smiles)
    logging.info(f"Fetched and cached {len(fetched_smiles)} SMILES.")
    
def fetch_cids_from_bioassay(bioassay_id):
    """
    Fetch Compound IDs (CIDs) from a given PubChem BioAssay ID.
    Returns a list of tuples: (CID, activity_label)
    activity_label: 1 for active, 0 for inactive
    """
    active_cids = []
    inactive_cids = []
    
    # Fetch Active CIDs
    active_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{bioassay_id}/activecid/JSON"
    try:
        response = requests.get(active_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        active_cids = data['InformationList']['Information'][0]['CID']
    except Exception as e:
        logging.error(f"Error fetching active CIDs for BioAssay {bioassay_id}: {e}")
    
    # Fetch Inactive CIDs
    inactive_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{bioassay_id}/inactivecid/JSON"
    try:
        response = requests.get(inactive_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        inactive_cids = data['InformationList']['Information'][0]['CID']
    except Exception as e:
        logging.error(f"Error fetching inactive CIDs for BioAssay {bioassay_id}: {e}")
    
    # Combine CIDs with labels
    cids_labels = [(cid, 1) for cid in active_cids] + [(cid, 0) for cid in inactive_cids]
    logging.info(f"BioAssay {bioassay_id}: {len(active_cids)} active CIDs and {len(inactive_cids)} inactive CIDs fetched.")
    return cids_labels

def build_dataframe_from_bioassay(bioassay_id, cache):
    """
    Build a DataFrame containing SMILES and activity labels from a given BioAssay.
    Utilizes caching to avoid redundant SMILES fetching.
    """
    cids_labels = fetch_cids_from_bioassay(bioassay_id)
    if not cids_labels:
        logging.error(f"No CIDs fetched for BioAssay {bioassay_id}.")
        return pd.DataFrame(columns=['SMILES', 'activity'])
    
    cids = [cid for cid, label in cids_labels]
    labels = {cid: label for cid, label in cids_labels}
    
    # Fetch SMILES, utilizing cache
    fetch_smiles_for_cids(cids, cache=cache)
    
    data = []
    for cid, label in cids_labels:
        smiles = cache.get(str(cid))
        if smiles:
            data.append({'SMILES': smiles, 'activity': label})
    
    df = pd.DataFrame(data)
    logging.info(f"BioAssay {bioassay_id}: {len(df)} SMILES matched with activity labels out of {len(cids_labels)} CIDs.")
    return df

# ==========================
# Descriptor Calculation
# ==========================
def calculate_additional_descriptors(mol):
    descriptors = {
        'TPSA': Descriptors.TPSA(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'RingCount': Descriptors.RingCount(mol),
        'MolWeight': Descriptors.MolWt(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
    }
    # Ensure consistent order
    return pd.Series([descriptors[name] for name in DESCRIPTOR_NAMES], index=DESCRIPTOR_NAMES)

# ==========================
# SMILES to Graph Conversion
# ==========================
def smiles_to_graph(smiles, label, fingerprint_size=1034, scaler=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(f"Invalid SMILES encountered: {smiles}")
        return None

    try:
        mol = Chem.AddHs(mol)
    except Exception as e:
        logging.error(f"Error adding hydrogens to molecule {smiles}: {e}")
        return None

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
            atom.GetImplicitValence(),
            atom.GetMass(),
            atom.GetIsInRing(),
            atom.GetTotalValence()
        ])
    x = torch.tensor(atom_features_list, dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        bond_type = bond.GetBondTypeAsDouble()
        edge_attr.append([bond_type])
        edge_attr.append([bond_type])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0, 1), dtype=torch.float)

    # Use 1024 bits for Morgan fingerprint
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fingerprint_array = np.zeros((1024,), dtype=int)
    DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
    fingerprint_tensor = torch.tensor(fingerprint_array, dtype=torch.float)

    additional_descriptors = calculate_additional_descriptors(mol)

    # Optional: Scale the descriptors if scaler is provided
    if scaler:
        try:
            additional_descriptors = scaler.transform([additional_descriptors.values])[0]
        except Exception as e:
            logging.error(f"Error scaling descriptors for molecule {smiles}: {e}")
            additional_descriptors = [0] * len(DESCRIPTOR_NAMES)
    additional_tensor = torch.tensor(additional_descriptors, dtype=torch.float)

    fingerprint_tensor = torch.cat([fingerprint_tensor, additional_tensor], dim=0)  # Final size: 1034
    assert fingerprint_tensor.size(0) == fingerprint_size, f"Fingerprint tensor size mismatch: expected {fingerprint_size}, got {fingerprint_tensor.size(0)}"

    y = torch.tensor([label], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, fingerprint=fingerprint_tensor)
    return data

# ==========================
# Molecule Dataset Class
# ==========================
class MoleculeDataset(Dataset):
    def __init__(self, dataframe, fingerprint_size=1034, scaler=None):
        self.dataframe = dataframe
        self.fingerprint_size = fingerprint_size
        self.scaler = scaler
        # Filter out invalid SMILES during initialization
        initial_count = len(self.dataframe)
        self.dataframe = self.dataframe[self.dataframe['SMILES'].apply(lambda x: Chem.MolFromSmiles(x) is not None)].reset_index(drop=True)
        filtered_count = len(self.dataframe)
        if filtered_count < initial_count:
            logging.warning(f"Filtered out {initial_count - filtered_count} invalid SMILES.")
        super(MoleculeDataset, self).__init__()

    def __len__(self):
        return len(self.dataframe)

    def get(self, idx):
        row = self.dataframe.iloc[idx]
        data = smiles_to_graph(row['SMILES'], row['activity'], self.fingerprint_size, scaler=self.scaler)
        if data is None:
            # This should not occur as we filtered invalid SMILES, but added as a safeguard
            data = Data(
                x=torch.empty((0, 10), dtype=torch.float),  # Updated for additional atom features
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1), dtype=torch.float),
                y=torch.tensor([row['activity']], dtype=torch.float),
                fingerprint=torch.zeros(self.fingerprint_size, dtype=torch.float)
            )
            logging.warning(f"Returned empty graph for SMILES: {row['SMILES']}")
        return data

    def __getitem__(self, idx):
        return self.get(idx), idx  # Return data and its index

# ==========================
# Custom Global Attention Layer
# ==========================
class CustomGlobalAttention(GlobalAttention):
    def __init__(self, gate_nn):
        super(CustomGlobalAttention, self).__init__(gate_nn)
        self.attn_weights_per_sample = {}  # Dict to store attention weights per sample
        self.max_keep = 1000  # Adjust based on memory constraints

    def forward(self, x, batch, indices=None):
        """
        Forward pass with sample indices to map attention weights.
        """
        gate = self.gate_nn(x)
        gate = torch.sigmoid(gate)
        self.all_attn_weights = gate.detach().cpu().numpy()
        x = x * gate
        out = scatter(x, batch, dim=0, reduce=self.aggr)
        
        if indices is not None:
            for i, idx in enumerate(indices):
                # Determine which nodes belong to the current sample
                sample_nodes = (batch == i).nonzero(as_tuple=False).squeeze().tolist()
                if isinstance(sample_nodes, int):
                    sample_nodes = [sample_nodes]
                attn_scores = gate[sample_nodes].cpu().numpy()
                self.attn_weights_per_sample[idx] = attn_scores
                # Limit the size of the dictionary to prevent memory issues
                if len(self.attn_weights_per_sample) > self.max_keep:
                    oldest_key = next(iter(self.attn_weights_per_sample))
                    del self.attn_weights_per_sample[oldest_key]
        return out

# ==========================
# Directed Message Passing Neural Network
# ==========================
class DirectedMPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(DirectedMPNN, self).__init__(aggr='add')  # "Add" aggregation.
        self.linear = nn.Linear(in_channels + 1, out_channels)
        self.update_layer = nn.Linear(out_channels, out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        num_nodes = x.size(0)
        edge_attr = edge_attr if edge_attr is not None else torch.zeros((edge_index.size(1), 1), device=edge_index.device)
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        loop_attr = torch.zeros((num_nodes, edge_attr.size(1)), device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        msg = torch.cat([x_j, edge_attr], dim=1)
        msg = self.linear(msg)
        msg = self.activation(msg)
        return msg

    def update(self, aggr_out):
        aggr_out = self.update_layer(aggr_out)
        aggr_out = self.activation(aggr_out)
        return aggr_out

# ==========================
# D-MPNN Model Definition
# ==========================
class DMPNN(nn.Module):
    def __init__(self, num_node_features, num_edge_features, fingerprint_size=1034, hidden_dim=256, num_layers=4):
        super(DMPNN, self).__init__()
        self.convs = nn.ModuleList([DirectedMPNN(num_node_features, hidden_dim) for _ in range(num_layers)])
        self.att_pool = CustomGlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        ))
        self.fc1 = nn.Linear(hidden_dim + fingerprint_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.att_weights = None

    def forward(self, data, indices=None):
        x, edge_index, edge_attr, fingerprint, batch = data.x, data.edge_index, data.edge_attr, data.fingerprint, data.batch
        fingerprint = fingerprint.to(x.device)  # Ensure fingerprint is on the same device
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        x = self.att_pool(x, batch, indices=indices)
        self.att_weights = self.att_pool.attn_weights_per_sample
        x = torch.cat([x, fingerprint], dim=1)  # Concatenate along feature dimension
        assert x.size(1) == self.fc1.in_features, f"Concatenated feature size mismatch: expected {self.fc1.in_features}, got {x.size(1)}"
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x).squeeze()

# ==========================
# Visualization Function
# ==========================
def visualize_attention(smiles, scores, threshold=0.05, size=(300, 300)):
    """
    Visualize attention scores on a molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES string.")
        return
    mol = Chem.AddHs(mol)
    num_atoms = mol.GetNumAtoms()
    
    scores = np.array(scores)
    if len(scores) != num_atoms:
        logging.error(f"Number of attention scores ({len(scores)}) does not match number of atoms ({num_atoms}).")
        print("Error: Mismatch between attention scores and atom count.")
        return
    
    score_sum = scores.sum()
    if score_sum == 0:
        norm_scores = np.zeros_like(scores)
    else:
        norm_scores = scores / (score_sum + 1e-6)
    
    max_score, min_score = norm_scores.max(), norm_scores.min()
    if max_score - min_score < 1e-6:
        norm_scores = np.zeros_like(norm_scores)
    else:
        norm_scores = (norm_scores - min_score) / (max_score - min_score + 1e-6)
    
    atom_colors = {
        atom.GetIdx(): (1.0, 0.0, 0.0) if norm_scores[atom.GetIdx()] > threshold else (0.5, 0.5, 0.5)
        for atom in mol.GetAtoms()
    }
    
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.atomColours = atom_colors
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    img = drawer.GetDrawingText()
    
    plt.figure(figsize=(4, 4))
    plt.imshow(plt.imread(io.BytesIO(img)))
    plt.axis('off')
    plt.show()

# ==========================
# Training Function
# ==========================
def train(model, loader, optimizer, criterion, device, grad_clip=None, accumulation_steps=1):
    """
    Train the model for one epoch.
    Supports gradient accumulation.
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, (data, indices) in enumerate(loader):
        data = data.to(device)
        out = model(data, indices=indices)
        loss = criterion(out, data.y) / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
    return total_loss / len(loader.dataset)

# ==========================
# Evaluation Function
# ==========================
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on the validation or test set.
    Returns loss, AUC, accuracy, precision, recall, F1 score, and confusion matrix.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, indices in loader:
            data = data.to(device)
            out = model(data, indices=indices)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(data.y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = 0.0
    # Additional metrics
    binary_preds = (np.array(all_preds) > 0.5).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    cm = confusion_matrix(all_labels, binary_preds)
    return avg_loss, auc, accuracy, precision, recall, f1, cm

# ==========================
# Training Loop with Early Stopping
# ==========================
def train_model(model, train_loader, test_loader, device, num_epochs=100, patience=10, criterion=None, grad_clip=1.0, accumulation_steps=1):
    """
    Train the model with early stopping based on validation AUC.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    best_auc = 0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, grad_clip=grad_clip, accumulation_steps=accumulation_steps)
        val_loss, val_auc, val_acc, val_prec, val_rec, val_f1, val_cm = evaluate(model, test_loader, criterion, device)
        logging.info(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
                     f'Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}, Val Prec={val_prec:.4f}, '
                     f'Val Rec={val_rec:.4f}, Val F1={val_f1:.4f}')
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
              f'Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}, Val Prec={val_prec:.4f}, '
              f'Val Rec={val_rec:.4f}, Val F1={val_f1:.4f}')

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            torch.save(best_model_state, 'best_dmpnn_model.pth')
            logging.info(f'New best model saved with AUC: {best_auc:.4f}')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info('Early stopping triggered')
            print('Early stopping triggered')
            break

    return best_auc

# ==========================
# Prediction Function
# ==========================
def predict(model, loader, device, threshold=0.5):
    """
    Generate predictions for the dataset.
    Returns a list of predicted probabilities.
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for data, indices in loader:
            data = data.to(device)
            out = model(data, indices=indices)
            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(preds)
    return all_preds

# ==========================
# Main Function
# ==========================
def main():
    # ==========================
    # Reproducibility Settings
    # ==========================
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ==========================
    # Load SMILES Cache
    # ==========================
    cache = load_cache()

    # ==========================
    # Data Retrieval from PubChem
    # ==========================
    # Define the BioAssay ID(s) you are interested in
    # For demonstration, we'll use BioAssay ID 1000 (you can replace it with your target BioAssay IDs)
    bioassay_ids = [1000]  # Replace with your BioAssay IDs

    all_data = []
    for bioassay_id in bioassay_ids:
        logging.info(f'Fetching data for BioAssay ID: {bioassay_id}')
        df = build_dataframe_from_bioassay(bioassay_id, cache)
        logging.info(f'Fetched {len(df)} records from BioAssay ID: {bioassay_id}')
        all_data.append(df)
        time.sleep(1)  # Sleep to respect rate limits

    if not all_data:
        logging.error('No data fetched from any BioAssay.')
        print('Error: No data fetched from any BioAssay.')
        return

    # Combine data from all BioAssays
    data_df = pd.concat(all_data, ignore_index=True)

    if data_df.empty:
        logging.error('Combined DataFrame is empty.')
        print('Error: Combined DataFrame is empty.')
        return

    # ==========================
    # Data Preprocessing
    # ==========================
    # Drop duplicates and NaNs
    initial_count = len(data_df)
    data_df = data_df.drop_duplicates(subset=['SMILES']).dropna(subset=['SMILES', 'activity'])
    filtered_count = len(data_df)
    if filtered_count < initial_count:
        logging.warning(f"Dropped {initial_count - filtered_count} duplicate or NaN SMILES.")
    data_df['activity'] = data_df['activity'].astype(int)

    # ==========================
    # Train-Test Split
    # ==========================
    train_df, test_df = train_test_split(
        data_df, test_size=0.2, random_state=42, stratify=data_df['activity']
    )
    logging.info(f'Training samples: {len(train_df)}, Testing samples: {len(test_df)}')

    # ==========================
    # Feature Scaling
    # ==========================
    scaler = StandardScaler()
    # Fit scaler on training descriptors only to prevent data leakage
    train_descriptors = train_df.apply(
        lambda row: calculate_additional_descriptors(Chem.MolFromSmiles(row['SMILES'])),
        axis=1
    )
    scaler.fit(train_descriptors.values)
    logging.info('StandardScaler fitted on training descriptors.')

    # ==========================
    # Dataset and DataLoader
    # ==========================
    fingerprint_size = 1034  # 1024 bits + 10 descriptors
    train_dataset = MoleculeDataset(train_df, fingerprint_size=fingerprint_size, scaler=scaler)
    test_dataset = MoleculeDataset(test_df, fingerprint_size=fingerprint_size, scaler=scaler)

    batch_size = 8  # Set batch size to 8
    num_workers = 4  # Adjust based on your system's capabilities
    pin_memory = True if torch.cuda.is_available() else False

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    # ==========================
    # Device Configuration
    # ==========================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    print(f'Using device: {device}')

    # ==========================
    # Class Imbalance Handling
    # ==========================
    num_pos = (train_df['activity'] == 1).sum()
    num_neg = (train_df['activity'] == 0).sum()
    if num_pos == 0:
        logging.error("No positive samples in the training data.")
        print("Error: No positive samples in the training data.")
        return
    pos_weight_value = num_neg / num_pos
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float).to(device)
    logging.info(f'Positional weight for BCEWithLogitsLoss: {pos_weight_value:.4f}')

    # ==========================
    # Model Initialization
    # ==========================
    num_node_features = 10  # Updated to match additional atom features
    num_edge_features = 1
    hidden_dim = 256  # Increased hidden dimension for better capacity
    num_layers = 4  # Increased number of layers for deeper representation

    model = DMPNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        fingerprint_size=fingerprint_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    logging.info('Model initialized.')

    # ==========================
    # Loss Function
    # ==========================
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logging.info('Loss function set to BCEWithLogitsLoss with pos_weight.')

    # ==========================
    # Training the Model
    # ==========================
    best_auc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=100,
        patience=10,
        criterion=criterion,
        grad_clip=1.0,  # Gradient clipping to prevent exploding gradients
        accumulation_steps=1  # Set to >1 for gradient accumulation
    )
    print(f'Best Validation AUC: {best_auc:.4f}')
    logging.info(f'Best Validation AUC: {best_auc:.4f}')

    # ==========================
    # Save Updated Cache
    # ==========================
    save_cache(cache)

    # ==========================
    # Loading Best Model
    # ==========================
    try:
        model.load_state_dict(torch.load('best_dmpnn_model.pth'))
        logging.info('Best model loaded from best_dmpnn_model.pth')
    except FileNotFoundError:
        logging.error('best_dmpnn_model.pth not found.')
        print('Error: best_dmpnn_model.pth not found.')
        return

    # ==========================
    # Testing the Model
    # ==========================
    test_loss, test_auc, test_acc, test_prec, test_rec, test_f1, test_cm = evaluate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}')
    print(f'Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1 Score: {test_f1:.4f}')
    print(f'Confusion Matrix:\n{test_cm}')
    logging.info(f'Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}, '
                 f'Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1 Score: {test_f1:.4f}')
    logging.info(f'Confusion Matrix:\n{test_cm}')

    # ==========================
    # Making Predictions
    # ==========================
    predictions = predict(model, test_loader, device, threshold=0.5)
    test_df = test_df.copy()
    test_df['Predicted_Activity_Prob'] = predictions
    test_df['Predicted_Activity'] = (np.array(predictions) > 0.5).astype(int)
    test_df.to_csv('test_predictions.csv', index=False)
    logging.info('Predictions saved to test_predictions.csv')
    print('Predictions saved to test_predictions.csv')

    # ==========================
    # Example Visualization for Multiple Molecules (Optional)
    # ==========================
    num_samples_to_visualize = 5  # Adjust as needed
    for idx in range(min(num_samples_to_visualize, len(test_df))):
        sample_smiles = test_df.iloc[idx]['SMILES']
        sample_data = smiles_to_graph(sample_smiles, test_df.iloc[idx]['activity'], fingerprint_size=fingerprint_size, scaler=scaler)
        if sample_data is not None:
            sample_data = sample_data.to(device)
            model.eval()
            with torch.no_grad():
                _ = model(sample_data, indices=[idx])
                attention_scores = model.att_weights.get(idx, [])
            if attention_scores.size > 0:
                visualize_attention(sample_smiles, attention_scores)
                logging.info(f'Attention visualization generated for SMILES: {sample_smiles}')
            else:
                logging.warning(f'No attention scores available for SMILES: {sample_smiles}')
        else:
            logging.warning('Sample molecule could not be converted to graph for visualization.')

if __name__ == '__main__':
    main()
