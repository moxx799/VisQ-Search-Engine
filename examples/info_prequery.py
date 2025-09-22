import pickle
import os
import numpy as np
import os
import faiss
from clustercontrast.utils.infomap_cluster_huang import  knn_faiss, knns2ordered_nbrs,get_links
import re
from collections import defaultdict
import networkx as nx
from scipy.sparse import csr_matrix
import json
import csv
import infomap
from tqdm import tqdm
import pickle
from clustercontrast.utils.infomap_utils import Timer
import pandas as pd
import operator

def tree_to_csv(tree_path, metadata_npz_path, output_csv):
    """
    Processes the Infomap tree file and metadata to create a CSV with coordinates and community paths.
    
    Parameters:
    tree_path (str): Path to the Infomap tree.tree file.
    metadata_npz_path (str): Path to the metadata .npz file.
    output_csv (str): Path where the output CSV will be saved.
    """
    # Parse the tree.tree file
    node_data = []
    with open(tree_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue  # Skip malformed lines
            path = parts[0]
            node_id = int(parts[3])
            node_data.append({'node_id': node_id, 'path': path})
    
    tree_df = pd.DataFrame(node_data).set_index('node_id')  # Set index

    # Load metadata from .npz file
    data = np.load(metadata_npz_path)
    meta_df = pd.DataFrame({
        'x': data['x'],
        'y': data['y']})
    # }, index=data['index'])  # Set index

    # Merge using index
    merged_df = tree_df.join(meta_df, how='left')  

    # Save to CSV
    #merged_df.to_csv(output_csv, index=True)  # Keeps index in CSV
    #print(f"Successfully saved merged data to {output_csv}")
    return merged_df

def split_hierarchy_levels(df):
    """
    Split the 'path' column into individual hierarchy level columns.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with 'path' column
    
    Returns:
    pd.DataFrame: DataFrame with new level columns
    """
    # Split path into separate components
    levels = df['path'].str.split(':', expand=True)
    
    # Rename columns to level_1, level_2, etc.
    levels = levels.rename(columns=lambda x: f'level_{x+1}')
    
    # Combine with original DataFrame
    return pd.concat([df, levels], axis=1)

def filter_small_modules(df, min_size=5):
    """
    Replace module numbers with -1 in level columns where the module contains fewer than min_size nodes.
    Operates on level columns (level_1, level_2, etc.) while maintaining hierarchy consistency.
    
    Parameters:
    df (pd.DataFrame): DataFrame with level columns
    min_size (int): Minimum number of nodes required to keep a module
    
    Returns:
    pd.DataFrame: Modified DataFrame with small modules marked as -1
    """
    # Get list of level columns
    level_cols = [col for col in df.columns if col.startswith('level_')]
    
    for col in level_cols:
        # Calculate module sizes
        module_counts = df.groupby(col)[col].transform('count')
        
        # Create mask for small modules
        small_modules_mask = module_counts < min_size
        
        # Replace small modules with -1, maintaining original data type
        df[col] = df[col].mask(small_modules_mask, -1)
    
    return df

class KNNGraphBuilder:
    """
    Constructs a K-Nearest Neighbor graph using FAISS, incorporating both feature similarity and location similarity.
    """
    def __init__(self, feature_dicts, weights=None, k=24, knn_method='faiss-gpu'):

    #     self.keys = list(feature_dict.keys())
    #     self.features = np.array([feature_dict[k].cpu().numpy() for k in self.keys]) # N x D
    #     self.knn_method = knn_method
    #     # Create lookup dictionaries
    #     self.metadata = [self._parse_filename(k) for k in self.keys]
    #     self.locs = None
    #     self.loc_weight = loc_weight
    #     self.k = k
    #     self.knns = None
    #     self.links = None
    #     self.single_nodes = []
          # Convert to list if single dict passed

        # Process feature dicts to use basename keys
        
        self.features = np.array([v for _, v in feature_dicts.items()])

        self.basenames = [ os.path.basename(k) for k in feature_dicts.keys()]
        self.weights = weights
        # Initialize remaining parameters
        self.k = k
        self.knn_method = knn_method
        self.metadata = [self._parse_filename(bn) for bn in self.basenames]
        self.keys = self.basenames
        
    def _parse_filename(self, filename):
        pattern = r'_c(\d+)_(\d+)s1_(\d+)\.tif$'
        match = re.search(pattern, filename)
        if match:
            return {
                'x': int(match.group(1)),
                'y': int(match.group(2)),
                # 'index': int(match.group(3)),
                #'original_key': filename
            }
        raise ValueError(f"Invalid filename format: {filename}")
    
    def save_metadata(self,path):
        np.savez_compressed(
            os.path.join(path, 'metadata.npz'),
            x=[m['x'] for m in self.metadata],
            y=[m['y'] for m in self.metadata],
            # index=[m['index'] for m in self.metadata],
            keys=self.keys
        )
    def precompute(self,):
        self.index = knn_faiss(feats=self.features,k=self.k, knn_method=self.knn_method)
        knns = self.index.get_knns()
        dists, nbrs = knns2ordered_nbrs(knns)
        
        return dists, nbrs
    
class CrossKNNGraphBuilder:
    """
    Constructs a KNN graph combining multiple feature sets with different paths but matching filenames.
    """
    def __init__(self, feature_dicts, k=24, knn_method='faiss-gpu'):
        # Convert to list if single dict passed
        if not isinstance(feature_dicts, list):
            feature_dicts = [feature_dicts]
            
        # Process feature dicts to use basename keys
        self.feature_sets = []
        self.features = []
        for i,feat_dict in enumerate(feature_dicts):
            # add i to the basename
            basename_dict = {f"{i}_{os.path.basename(k)}": v for k, v in feat_dict.items()}
            self.feature_sets.append(basename_dict)
            self.features.append(np.array([v for _, v in feat_dict.items()]))
        self.basenames = list(self.feature_sets[0].keys())
        print('the length of the basenames is:', len(self.basenames))

        
        # Initialize remaining parameters
        self.k = k
        self.knn_method = knn_method
        self.metadata = [self._parse_filename(bn) for bn in self.basenames]
        self.index = None

    def _parse_filename(self, basename):
        """Parse metadata from filename basename"""
        pattern = r'_c(\d+)_(\d+)s1_(\d+)\.tif$'
        match = re.search(pattern, basename)
        if match:
            return {
                'x': int(match.group(1)),
                'y': int(match.group(2)),
            }
        raise ValueError(f"Invalid filename format: {basename}")

    def save_metadata(self, path):
        """Save metadata using original basenames as keys"""
        np.savez_compressed(
            os.path.join(path, 'metadata.npz'),
            x=[m['x'] for m in self.metadata],
            y=[m['y'] for m in self.metadata],
            keys=self.basenames
        )

    def precompute(self):
        """FAISS KNN search with combined features"""
        self.index = knn_faiss(feats=self.features, k=self.k, knn_method=self.knn_method)
        knns = self.index.get_knns()
        return knns2ordered_nbrs(knns)
    

class MultiKNNGraphBuilder:
    """
    Constructs a KNN graph combining multiple feature sets with different paths but matching filenames.
    """
    def __init__(self, feature_dicts, weights=None, k=24, knn_method='faiss-gpu'):
        # Convert to list if single dict passed
        if not isinstance(feature_dicts, list):
            feature_dicts = [feature_dicts]
            
        # Process feature dicts to use basename keys
        self.feature_sets = []
        for feat_dict in feature_dicts:
            basename_dict = {os.path.basename(k): v for k, v in feat_dict.items()}
            self.feature_sets.append(basename_dict)
        # print one example of the feature dict
        print(list(self.feature_sets[0].keys())[0])
        # Validate consistent samples across feature sets
        self.basenames = list(self.feature_sets[0].keys())
        self.weights = weights
        if len(self.weights) != len(feature_dicts):
            raise ValueError("Weights count must match feature set count")
            
        # Combine features using weighted concatenation
        self.features = self._combine_features()
        
        # Initialize remaining parameters
        self.k = k
        self.knn_method = knn_method
        self.metadata = [self._parse_filename(bn) for bn in self.basenames]
        self.index = None

    def _parse_filename(self, basename):
        """Parse metadata from filename basename"""
        pattern = r'_c(\d+)_(\d+)s1_(\d+)\.tif$'
        match = re.search(pattern, basename)
        if match:
            return {
                'x': int(match.group(1)),
                'y': int(match.group(2)),
            }
        raise ValueError(f"Invalid filename format: {basename}")

    def _combine_features(self):
        """Combine features from different sets with weight scaling"""
        combined = []
        for bn in self.basenames:
            scaled_features = []
            for weight, feat_dict in zip(self.weights, self.feature_sets):
                feat = feat_dict[bn].cpu().numpy()
                scaled_features.append(feat * np.sqrt(weight))
            combined.append(np.concatenate(scaled_features))
        return np.array(combined)

    def save_metadata(self, path):
        """Save metadata using original basenames as keys"""
        np.savez_compressed(
            os.path.join(path, 'metadata.npz'),
            x=[m['x'] for m in self.metadata],
            y=[m['y'] for m in self.metadata],
            keys=self.basenames
        )
    def combine_or(self,nbrs1, dists1, nbrs2, dists2, k=None):
        n_nodes = nbrs1.shape[0]
        or_nbrs = []
        or_dists = []
        for i in range(n_nodes):
            neighbors = {}
            # Process first KNN
            for j, d in zip(nbrs1[i], dists1[i]):
                if j not in neighbors or d < neighbors.get(j, np.inf):
                    neighbors[j] = d
            # Process second KNN
            for j, d in zip(nbrs2[i], dists2[i]):
                if j not in neighbors or d < neighbors.get(j, np.inf):
                    neighbors[j] = d
            # Sort by distance
            sorted_items = sorted(neighbors.items(), key=lambda x: x[1])
            # Truncate to k if specified
            if k is not None:
                sorted_items = sorted_items[:k]
            sorted_nbrs = [item[0] for item in sorted_items]
            sorted_dists = [item[1] for item in sorted_items]
            # Pad if needed
            if k is not None and len(sorted_nbrs) < k:
                pad = k - len(sorted_nbrs)
                sorted_nbrs += [-1] * pad
                sorted_dists += [np.inf] * pad
            or_nbrs.append(np.array(sorted_nbrs, dtype=np.int32))
            or_dists.append(np.array(sorted_dists))
        # Convert to numpy arrays
        return np.stack(or_dists), np.stack(or_nbrs)

    def combine_and(self,nbrs1, dists1, nbrs2, dists2, k=None, weight=0.5):
        n_nodes = nbrs1.shape[0]
        and_nbrs = []
        and_dists = []
        combine_func = combine_func or (lambda x, y: weight*x +(1-weight)* y)
        for i in range(n_nodes):
            set1 = set(nbrs1[i])
            set2 = set(nbrs2[i])
            common = set1.intersection(set2)
            index1 = {j: idx for idx, j in enumerate(nbrs1[i])}
            index2 = {j: idx for idx, j in enumerate(nbrs2[i])}
            combined = {}
            for j in common:
                d1 = dists1[i][index1[j]]
                d2 = dists2[i][index2[j]]
                combined[j] = combine_func(d1, d2)
            sorted_items = sorted(combined.items(), key=lambda x: x[1])
            # Truncate to k if specified
            if k is not None:
                sorted_items = sorted_items[:k]
            sorted_nbrs = [item[0] for item in sorted_items]
            sorted_dists = [item[1] for item in sorted_items]
            # Pad if needed
            if k is not None and len(sorted_nbrs) < k:
                pad = k - len(sorted_nbrs)
                sorted_nbrs += [-1] * pad
                sorted_dists += [np.inf] * pad
            and_nbrs.append(np.array(sorted_nbrs, dtype=np.int32))
            and_dists.append(np.array(sorted_dists))
        # Convert to numpy arrays
        return np.stack(and_dists), np.stack(and_nbrs)
        
    def _combine_features(self):
        """Combine features from different sets with weight scaling"""
        combined = []
        for bn in self.basenames:
            scaled_features = []
            for weight, feat_dict in zip(self.weights, self.feature_sets):
                feat = feat_dict[bn].cpu().numpy()
                scaled_features.append(feat * np.sqrt(weight))
            combined.append(np.concatenate(scaled_features))
        return np.array(combined)
    
    def precompute(self):
        """FAISS KNN search with combined features"""
        self.index = knn_faiss(feats=self.features, k=self.k, knn_method=self.knn_method)
        knns = self.index.get_knns()
        return knns2ordered_nbrs(knns)
    
    
class CleanMultiKNNGraphBuilder:
    """
    Constructs a KNN graph combining multiple feature sets with different paths but matching filenames.
    """
    def __init__(self, feature_dicts, weights=None, k=24, knn_method='faiss-gpu'):
        # Convert to list if single dict passed
        if not isinstance(feature_dicts, list):
            feature_dicts = [feature_dicts]
        # Process feature dicts to use basename keys
        self.feature_sets = []
        for feat_dict in feature_dicts:
            basename_dict = {os.path.basename(k): v for k, v in feat_dict.items()}
            self.feature_sets.append(basename_dict)
        # print one example of the feature dict
        print(list(self.feature_sets[0].keys())[0])
        # Validate consistent samples across feature sets
        self.basenames = list(self.feature_sets[0].keys())
        self.weights = weights
        if len(self.weights) != len(feature_dicts):
            raise ValueError("Weights count must match feature set count")
        self.k = k
        self.knn_method = knn_method
        self.metadata = [self._parse_filename(bn) for bn in self.basenames]
        self.index = None
        self.features = self._combine_features()

    def _parse_filename(self, basename):
        """Parse metadata from filename basename"""
        pattern = r'_c(\d+)_(\d+)s1_(\d+)\.tif$'
        match = re.search(pattern, basename)
        if match:
            return {
                'x': int(match.group(1)),
                'y': int(match.group(2)),
            }
        raise ValueError(f"Invalid filename format: {basename}")


    def save_metadata(self, path):
        """Save metadata using original basenames as keys"""
        np.savez_compressed(
            os.path.join(path, 'metadata.npz'),
            x=[m['x'] for m in self.metadata],
            y=[m['y'] for m in self.metadata],
            keys=self.basenames
        )
    def _combine_features(self):
        """Combine features from different sets with weight scaling"""
        combined = []
        for bn in self.basenames:
            scaled_features = []
            for weight, feat_dict in zip(self.weights, self.feature_sets):
                feat = feat_dict[bn].cpu().numpy()
                scaled_features.append(feat * np.sqrt(weight))
            combined.append(np.concatenate(scaled_features))
        return np.array(combined)    
        
    def precompute(self):
        """FAISS KNN search with combined features"""
        self.index = knn_faiss(feats=self.features, k=self.k, knn_method=self.knn_method)
        knns = self.index.get_knns()
        return knns2ordered_nbrs(knns)
    
    
    

    
class CombinKNNGraphBuilder:
    """
    Constructs a KNN graph combining multiple feature sets with different paths but matching filenames.
    """
    def __init__(self, feature_dicts, weights=None, k=24, knn_method='faiss-gpu'):
        # Convert to list if single dict passed
        if not isinstance(feature_dicts, list):
            feature_dicts = [feature_dicts]
            
        # Process feature dicts to use basename keys
        self.feature_sets = []
        for feat_dict in feature_dicts:
            basename_dict = {os.path.basename(k): v for k, v in feat_dict.items()}
            self.feature_sets.append(basename_dict)
        # print one example of the feature dict
        print(list(self.feature_sets[0].keys())[0])
        # Validate consistent samples across feature sets
        self.basenames = list(self.feature_sets[0].keys())
        self.weights = weights
        if len(self.weights) != len(feature_dicts):
            raise ValueError("Weights count must match feature set count")

        # Initialize remaining parameters
        self.k = k
        self.knn_method = knn_method
        self.metadata = [self._parse_filename(bn) for bn in self.basenames]
        self.index = None
        
        
    def expert_wise(self,):
        feat_dists = []
        feat_nbrs = []
        for feature in self.feature_sets:
            self.index = knn_faiss(feats=feature, k=self.k, knn_method=self.knn_method)
            knns = self.index.get_knns()
            feat_dist, feat_nbr = knns2ordered_nbrs(knns)
            feat_dists.append(feat_dist)
            feat_nbrs.append(feat_nbr)
        return feat_dists, feat_nbrs
        

    def _parse_filename(self, basename):
        """Parse metadata from filename basename"""
        pattern = r'_c(\d+)_(\d+)s1_(\d+)\.tif$'
        match = re.search(pattern, basename)
        if match:
            return {
                'x': int(match.group(1)),
                'y': int(match.group(2)),
            }
        raise ValueError(f"Invalid filename format: {basename}")


    def save_metadata(self, path):
        """Save metadata using original basenames as keys"""
        np.savez_compressed(
            os.path.join(path, 'metadata.npz'),
            x=[m['x'] for m in self.metadata],
            y=[m['y'] for m in self.metadata],
            keys=self.basenames
        )
    def combine_or(self,nbrs1, dists1, nbrs2, dists2, k=None):
        n_nodes = nbrs1.shape[0]
        or_nbrs = []
        or_dists = []
        for i in range(n_nodes):
            neighbors = {}
            # Process first KNN
            for j, d in zip(nbrs1[i], dists1[i]):
                if j not in neighbors or d < neighbors.get(j, np.inf):
                    neighbors[j] = d
            # Process second KNN
            for j, d in zip(nbrs2[i], dists2[i]):
                if j not in neighbors or d < neighbors.get(j, np.inf):
                    neighbors[j] = d
            # Sort by distance
            sorted_items = sorted(neighbors.items(), key=lambda x: x[1])
            # Truncate to k if specified
            if k is not None:
                sorted_items = sorted_items[:k]
            sorted_nbrs = [item[0] for item in sorted_items]
            sorted_dists = [item[1] for item in sorted_items]
            # Pad if needed
            if k is not None and len(sorted_nbrs) < k:
                pad = k - len(sorted_nbrs)
                sorted_nbrs += [-1] * pad
                sorted_dists += [np.inf] * pad
            or_nbrs.append(np.array(sorted_nbrs, dtype=np.int32))
            or_dists.append(np.array(sorted_dists))
        # Convert to numpy arrays
        return np.stack(or_dists), np.stack(or_nbrs)

    def combine_and(self,nbrs1, dists1, nbrs2, dists2, k=None, ):
        weight=self.weights[0]
        n_nodes = nbrs1.shape[0]
        and_nbrs = []
        and_dists = []
        combine_func = combine_func or (lambda x, y: weight*x +(1-weight)* y)
        for i in range(n_nodes):
            set1 = set(nbrs1[i])
            set2 = set(nbrs2[i])
            common = set1.intersection(set2)
            index1 = {j: idx for idx, j in enumerate(nbrs1[i])}
            index2 = {j: idx for idx, j in enumerate(nbrs2[i])}
            combined = {}
            for j in common:
                d1 = dists1[i][index1[j]]
                d2 = dists2[i][index2[j]]
                combined[j] = combine_func(d1, d2)
            sorted_items = sorted(combined.items(), key=lambda x: x[1])
            # Truncate to k if specified
            if k is not None:
                sorted_items = sorted_items[:k]
            sorted_nbrs = [item[0] for item in sorted_items]
            sorted_dists = [item[1] for item in sorted_items]
            # Pad if needed
            if k is not None and len(sorted_nbrs) < k:
                pad = k - len(sorted_nbrs)
                sorted_nbrs += [-1] * pad
                sorted_dists += [np.inf] * pad
            and_nbrs.append(np.array(sorted_nbrs, dtype=np.int32))
            and_dists.append(np.array(sorted_dists))
        # Convert to numpy arrays
        return np.stack(and_dists), np.stack(and_nbrs)
        

    def precompute(self):
        """FAISS KNN search with combined features"""
        self.index = knn_faiss(feats=self.features, k=self.k, knn_method=self.knn_method)
        knns = self.index.get_knns()
        return knns2ordered_nbrs(knns)
    
    def merge(self):
        feat_dists, feat_nbrs = self.expert_wise()
        return self.combine_and(feat_nbrs[0], feat_dists[0], feat_nbrs[1], feat_dists[1])
        

class CommunityDetector:
    """
    Detects hierarchical communities using Infomap and handles querying by location.
    """
    def __init__(self, metadata_path=None):
        data = np.load(metadata_path, allow_pickle=True)

        self.metadata = [
            {'x': x, 'y': y}
            for x, y in zip(data['x'], data['y'],)
            # for x, y, idx in zip(data['x'], data['y'], data['index'])
        ]
        self.links = None
        self.single_nodes = []
        self.node_hierarchy = {}  # {node_id: [community_path]}
        self.community_structure = defaultdict(lambda: defaultdict(list))  # {level: {community_id: [nodes]}}

    def detect_hierarchical_communities(self, links, single_nodes,save_dir, save_ranked=False):
        """Run Infomap to detect hierarchical communities and structure."""
        self.links = links
        self.single_nodes = single_nodes
        im = infomap.Infomap("--directed --tree")
        print("Building network...")    
        for (i, j), sim in tqdm(self.links.items()):
            _ = im.addLink(int(i), int(j), sim)
        # for (i, j), sim in self.links.items():
        #      _ = im.addLink(int(i), int(j), sim)
        print(f"Detected {len(self.links)} links.")
        im.run()
        im.writeTree(os.path.join(save_dir, 'tree.tree'))
        # Extract hierarchical path using .path attribute
        self.node_hierarchy = {}
  
        for node in im.tree:  # Iterate through all nodes
            
            hierarchy_path = str(node.module_id).split(':') 
          
            self.node_hierarchy[node.node_id] = hierarchy_path

        #Handle isolated nodes
        for node_id in single_nodes:

            if node_id not in self.node_hierarchy:
                self.node_hierarchy[node_id] = ["isolated"]
        del im
        
        # two level infomap
        im = infomap.Infomap("--directed --tree --two-level")
        for (i, j), sim in tqdm(self.links.items()):
            _ = im.addLink(int(i), int(j), sim)
        im.run()
        im.writeTree(os.path.join(save_dir, 'tree2.tree'))
        if save_ranked is True:
            node_to_community = {node.node_id: node.module_id for node in im.tree if node.is_leaf}
            # Step 3: Aggregate edge weights between communities
            community_weights = defaultdict(float)
            for (i, j), sim in links.items():
                comm_i = node_to_community.get(int(i), -1)
                comm_j = node_to_community.get(int(j), -1)
                if comm_i != -1 and comm_j != -1 and comm_i != comm_j:  # Ignore intra-community edges
                    community_weights[(comm_i, comm_j)] += sim

            # Step 4: For each community, rank other communities by total weight
            community_neighbors = defaultdict(list)
            for (comm_i, comm_j), weight in community_weights.items():
                community_neighbors[comm_i].append((comm_j, weight))

            # Step 5: Prepare output with top nearest communities
            output_data = {}
            for comm_i in community_neighbors:
                # Sort neighboring communities by total weight (descending)
                ranked_communities = sorted(
                    community_neighbors[comm_i], key=operator.itemgetter(1), reverse=True
                )
                # Take top k communities (or fewer if < k)
                top_communities = ranked_communities[:50]
                output_data[comm_i] = {
                    'neighbors': [
                        {'community': int(comm_j), 'weight': float(weight)}
                        for comm_j, weight in top_communities
                    ]
                }

            # Step 6: Save to JSON file
            if save_ranked:
                saved_path = os.path.join(save_dir, 'top_nearest_communities.json')
                with open(saved_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
            

    def save_hierarchy(self, save_dir):
        """Save metadata, community structure, and node hierarchy to disk."""
        os.makedirs(save_dir, exist_ok=True)
        # Save metadata
        with open(os.path.join(save_dir, 'metadata.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'x', 'y'])
            for meta in self.metadata:
                writer.writerow([meta['index'], meta['x'], meta['y'],])
        # Save node hierarchy
        with open(os.path.join(save_dir, 'node_hierarchy.json'), 'w') as f:
            json.dump({str(k): v for k, v in self.node_hierarchy.items()}, f)
        # Save community structure
        comm_structure = {level: dict(comms) for level, comms in self.community_structure.items()}
        with open(os.path.join(save_dir, 'community_structure.json'), 'w') as f:
            json.dump(comm_structure, f)

    def load_hierarchy(self, load_dir):
        """Load saved hierarchy data from disk."""
        # Load metadata
        self.metadata = []
        with open(os.path.join(load_dir, 'metadata.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.metadata.append({
                    'index': int(row['index']),
                    'x': int(row['x']),
                    'y': int(row['y']),
                    # 'original_key': row['original_key']
                })
        # Load node hierarchy
        with open(os.path.join(load_dir, 'node_hierarchy.json'), 'r') as f:
            self.node_hierarchy = {int(k): v for k, v in json.load(f).items()}
        # Load community structure
        with open(os.path.join(load_dir, 'community_structure.json'), 'r') as f:
            comm_structure = json.load(f)
            self.community_structure = defaultdict(lambda: defaultdict(list))
            for level_str, comms in comm_structure.items():
                level = int(level_str)
                self.community_structure[level] = defaultdict(list, comms)

    def query_by_location(self, x, y):
        """Query communities by location, returning all nodes in the same and higher-level communities."""
        target_indices = [meta['index'] for meta in self.metadata if meta['x'] == x and meta['y'] == y]
        if not target_indices:
            return {}
        # Collect all relevant community IDs across hierarchy levels
        relevant_communities = defaultdict(set)
        for idx in target_indices:
            if idx not in self.node_hierarchy:
                continue
            path = self.node_hierarchy[idx]
            for level in range(len(path)):
                comm_id = ':'.join(path[:level+1])
                relevant_communities[level].add(comm_id)
        # Aggregate all nodes in relevant communities
        result = defaultdict(list)
        for level, comm_ids in relevant_communities.items():
            for comm_id in comm_ids:
                result[level].extend(self.community_structure[level].get(comm_id, []))
        return dict(result)


def load_features(feature_paths) :

    features = []
    for path in feature_paths:
        file_path = os.path.abspath(path)  # Ensure absolute path for clarity
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Feature file not found: {file_path}")

        with open(file_path, 'rb') as f:
            features.append(pickle.load(f))

    return features


import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--feature_path', type=str, required=True, help='Path to the feature pickle file')
argparser.add_argument('--feature_path2', type=str, default=None, required=False, help='Path to the feature pickle file')
argparser.add_argument('--feature_path3', type=str, default=None, required=False, help='Path to the feature pickle file') 
argparser.add_argument(
    "--feature_paths",nargs="+",help="List of paths to feature files",required=False)
argparser.add_argument('--saved_path', type=str, required=True, help='Path to save the metadata and hierarchy')
argparser.add_argument('--weight1', type=float, default=0.5, help='Weight for the first feature set (if using two feature sets)')
argparser.add_argument('--mode', type=str, default=None, help='multi experts mode')
args = argparser.parse_args()
if args.mode is None:
    if args.feature_path2 is not None:
        feature_path = args.feature_path2
        saved_path = args.saved_path
        os.makedirs(saved_path, exist_ok=True)
        file_path = os.path.join(feature_path)
        with open(file_path, 'rb') as f:
            rawfeatures1 = pickle.load(f)
        feature_path = args.feature_path
        file_path = os.path.join(feature_path)
        with open(file_path, 'rb') as f:
            rawfeatures2 = pickle.load(f)
        graph_builder = MultiKNNGraphBuilder(feature_dicts=[rawfeatures1, rawfeatures2],weights=[args.weight1,1-args.weight1], k=24)
    else:

        feature_path = args.feature_path
        saved_path = args.saved_path
        os.makedirs(saved_path, exist_ok=True)
        # Specify the path to your saved file
        
        # Open and load the pickle file
        with open(feature_path, 'rb') as f:
            rawfeatures = pickle.load(f)
        graph_builder = KNNGraphBuilder(feature_dicts=rawfeatures, k=24)
        
elif args.mode == 'cross':
    assert args.feature_path2 is not None
    feature_path = args.feature_path2
    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)
    file_path = os.path.join(feature_path)
    with open(file_path, 'rb') as f:
        rawfeatures1 = pickle.load(f)
    feature_path = args.feature_path
    file_path = os.path.join(feature_path)
    with open(file_path, 'rb') as f:
        rawfeatures2 = pickle.load(f)
    graph_builder = CrossKNNGraphBuilder(feature_dicts=[rawfeatures1, rawfeatures2], k=24)

elif args.mode == 'and':
    assert args.feature_path2 is not None
    feature_path = args.feature_path2
    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)
    file_path = os.path.join(feature_path)
    with open(file_path, 'rb') as f:
        rawfeatures1 = pickle.load(f)
    feature_path = args.feature_path
    file_path = os.path.join(feature_path)
    with open(file_path, 'rb') as f:
        rawfeatures2 = pickle.load(f)
    graph_builder = MultiKNNGraphBuilder(feature_dicts=[rawfeatures1, rawfeatures2],weights=[args.weight1,1-args.weight1], k=24)
elif args.mode == '3':
        feature_path = args.feature_path2
        saved_path = args.saved_path
        os.makedirs(saved_path, exist_ok=True)
        file_path = os.path.join(feature_path)
        with open(file_path, 'rb') as f:
            rawfeatures1 = pickle.load(f)
        feature_path = args.feature_path
        file_path = os.path.join(feature_path)
        with open(file_path, 'rb') as f:
            rawfeatures2 = pickle.load(f)
        feature_path = args.feature_path3
        file_path = os.path.join(feature_path)
        with open(file_path, 'rb') as f:
            rawfeatures3 = pickle.load(f)
        
        graph_builder = MultiKNNGraphBuilder(feature_dicts=[rawfeatures1, rawfeatures2,rawfeatures3],weights=[0.67*args.weight1,(1-args.weight1)*0.67,0.33], k=24)   
        
elif args.mode == 'clean':
    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)
    feature_dicts = load_features(args.feature_paths)
    graph_builder = CleanMultiKNNGraphBuilder(feature_dicts = feature_dicts,weights=[0.1,0.35,0.35,0.1,0.1], k=24)
    
else:
    raise ValueError("Invalid mode. Choose from 'cross', 'and', '3', 'clean' or None for single feature set.")
    
feat_dists, feat_nbrs = graph_builder.precompute()
graph_builder.save_metadata(saved_path)
detector = CommunityDetector(metadata_path=os.path.join(saved_path, 'metadata.npz'))

single = []
links = {}
with Timer('get links', verbose=True):
    single, links = get_links(single=single, links=links, nbrs=feat_nbrs, dists=feat_dists, min_sim=0.5)    
detector.detect_hierarchical_communities(links, single,save_dir=saved_path, save_ranked=False)

# door = 'myelo_type_glial'
# df2 = tree_to_csv(f'{door}/tree2.tree', f'{door}/metadata.npz', f'{door}_2.csv')
# df2 = split_hierarchy_levels(df2)
# df2 = filter_small_modules(df2, min_size=5)
# df2['level_1'] = pd.to_numeric(df2['level_1'], errors='coerce')
# df2['level_2'] = pd.to_numeric(df2['level_2'], errors='coerce')
# df2.to_csv(f'{door}/{door}_2lv.csv', index=True) 

