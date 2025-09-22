import pandas as pd
import numpy as np

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

def main(args):
    door = args.door
    df2 = tree_to_csv(f'{door}/tree2.tree', f'{door}/metadata.npz', f'{door}_2.csv')
    df2 = split_hierarchy_levels(df2)
    df2 = filter_small_modules(df2, min_size=5)
    
    df2['level_1'] = pd.to_numeric(df2['level_1'], errors='coerce')
    df2['level_2'] = pd.to_numeric(df2['level_2'], errors='coerce')
    df2.to_csv(f'{door}/{door}_2lv.csv', index=True)  # Save the DataFrame to CSV
    
    df = tree_to_csv(f'{door}/tree.tree', f'{door}/metadata.npz', f'{door}.csv')
    df = split_hierarchy_levels(df)
    df = filter_small_modules(df, min_size=5)
    
    levs = []
    for col in df.columns:
        if col.startswith('level_'):
            levs.append(col)
    for col in levs:
       
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.to_csv(f'{door}/{door}_nlv.csv', index=True)  # Save the DataFrame to CSV

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Getting csv file for qupath")
    parser.add_argument('--door', type=str, default='cell_type_query')
    main(parser.parse_args())
