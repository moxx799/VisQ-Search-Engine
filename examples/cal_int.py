import torch
import tifffile
import pandas as pd
import numpy as np
import os

def process_boxes(input_df, tif_path,marker='marker', device=None):
    """
    Process bounding boxes to calculate min and std of expanded regions in a 16-bit TIFF image.
    
    Args:
        input_df (pd.DataFrame): DataFrame containing bounding box coordinates with columns
                                ['centroid_x', 'centroid_y', 'xmin', 'ymin', 'xmax', 'ymax']
        tif_path (str): Path to the 16-bit TIFF file
        output_csv (str): Path to save the output CSV file
        device (torch.device): PyTorch device (default: GPU if available, else CPU)
    
    Returns:
        pd.DataFrame: Processed DataFrame with new 'min_intensity' and 'std_intensity' columns
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Read TIFF image
    image = tifffile.imread(tif_path)
    
    # transfer uint16 to int8
    image = (image/ 256).astype(np.uint8)
    height, width = image.shape
    
    # Convert to PyTorch tensor and move to device
    image_tensor = torch.from_numpy(image).to(device)
    
    # Prepare output lists
    min_values = []
    std_values = []
    max_values = []
    
    # Process each bounding box
    for _, row in input_df.iterrows():
        # Convert coordinates to integers
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])
        
        # Expand coordinates with boundary checks
        new_xmin = max(0, xmin - 15)
        new_ymin = max(0, ymin - 15)
        new_xmax = min(width - 1, xmax + 15)
        new_ymax = min(height - 1, ymax + 15)
        
        # Skip invalid regions (shouldn't happen with proper inputs)
        if new_xmax <= new_xmin or new_ymax <= new_ymin:
            min_values.append(np.nan)
            std_values.append(np.nan)
            max_values.append(np.nan)
            continue
        
        # Extract region from tensor
        region = image_tensor[new_ymin:new_ymax+1, new_xmin:new_xmax+1]
        
        # Compute statistics
        mean_val = region.float().mean()  # Convert to float for stable mean calculation
        std_val = region.float().std()  # Convert to float for stable std calculation
        max_val = region.float().max()  # Convert to float for stable max calculation
        # Store results
        min_values.append(mean_val.cpu().item())
        std_values.append(std_val.cpu().item())
        max_values.append(max_val.cpu().item())
    # Create new DataFrame with results
    result_df = input_df.copy()
    result_df[f'{marker}_mean'] = min_values
    result_df[f'{marker}_std'] = std_values
    result_df[f'{marker}_max'] = max_values
    # Save to CSV
    
    
    return result_df

def process_175boxes(input_df, tif_path, marker='marker', device=None):
    """
    Process bounding boxes to calculate mean, std, and max of fixed-size regions (175x175) in a 16-bit TIFF image.
    
    Args:
        input_df (pd.DataFrame): DataFrame containing centroid coordinates with columns
                                ['centroid_x', 'centroid_y', ...]
        tif_path (str): Path to the 16-bit TIFF file
        marker (str): Prefix for the output columns (default: 'marker')
        device (torch.device): PyTorch device (default: GPU if available, else CPU)
    
    Returns:
        pd.DataFrame: Processed DataFrame with new columns for intensity statistics
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Read TIFF image
    image = tifffile.imread(tif_path)
    
    # Convert uint16 to uint8
    image = (image / 256).astype(np.uint8)
    height, width = image.shape
    
    # Convert to PyTorch tensor and move to device
    image_tensor = torch.from_numpy(image).to(device)
    
    # Prepare output lists
    mean_values = []
    std_values = []
    max_values = []
    
    # Calculate half-size for 175x175 box
    half_size = 175 // 2  # Results in 87
    
    # Process each row based on centroids
    for _, row in input_df.iterrows():
        # Get center coordinates and convert to integers
        cx = int(row['centroid_x'])
        cy = int(row['centroid_y'])
        
        # Calculate bounding box coordinates
        xmin = cx - half_size
        ymin = cy - half_size
        xmax = cx + half_size
        ymax = cy + half_size
        
        # Apply boundary checks
        new_xmin = max(0, xmin)
        new_ymin = max(0, ymin)
        new_xmax = min(width - 1, xmax)
        new_ymax = min(height - 1, ymax)
        
        # Check for invalid regions
        if new_xmax <= new_xmin or new_ymax <= new_ymin:
            mean_values.append(np.nan)
            std_values.append(np.nan)
            max_values.append(np.nan)
            continue
        
        # Extract region from tensor
        region = image_tensor[new_ymin:new_ymax+1, new_xmin:new_xmax+1]
        
        # Compute statistics
        region_float = region.float()
        mean_val = region_float.mean()
        std_val = region_float.std()
        max_val = region_float.max()
        
        # Store results
        mean_values.append(mean_val.cpu().item())
        std_values.append(std_val.cpu().item())
        max_values.append(max_val.cpu().item())
    
    # Create new DataFrame with results
    result_df = input_df.copy()
    result_df[f'{marker}_mean'] = mean_values
    result_df[f'{marker}_std'] = std_values
    result_df[f'{marker}_max'] = max_values
    
    return result_df

df = pd.read_csv('/home/lhuang37/datasets/50_plex/S1/classification_table_master.csv')
neuronal_connectivity_panel = {
    'NeuN': 'S1_R2C4.tif',
    'Neurofilament-H': 'S1_R5C5.tif',
    'Neurofilament-M': 'S1_R5C7.tif',
    'MAP2': 'S1_R5C9.tif',
    'Synaptophysin': 'S1_R2C5.tif',
    'CNPase': 'S1_R5C4.tif',
    'MBP': 'S1_R5C6.tif'
}
cell_type_panel = {
    'NeuN': 'S1_R2C4.tif',
    'Iba1': 'S1_R1C5.tif',
    'S100': 'S1_R3C5.tif',
    'Olig2': 'S1_R1C9.tif',
    'RECA1': 'S1_R1C6.tif'
}

glial_panel = {
    'MBP': 'S1_R5C6.tif',
    'GLAST': 'S1_R3C9.tif',
    'GFAP': 'S1_R3C3.tif',
    'CNPase': 'S1_R5C4.tif',
    'Sox2': 'S1_R4C5.tif',
    'S100': 'S1_R3C5.tif',
    'Olig2': 'S1_R1C9.tif',
    'Iba1': 'S1_R1C5.tif'
}

file_path = '/home/lhuang37/datasets/50_plex/S1/final'
# for marker, file_name in neuronal_connectivity_panel.items():
#     tif_path = os.path.join(file_path, file_name)
#     df = process_boxes(df, tif_path, marker=marker)
# output_csv = os.path.join('/home/lhuang37/repos/VisQ-Search-Engine/examples', 'marker_stats.csv')    
# df.to_csv(output_csv, index=False)


# for marker, file_name in cell_type_panel.items():
#     tif_path = os.path.join(file_path, file_name)
#     df = process_175boxes(df, tif_path, marker=marker)
# output_csv = os.path.join('/home/lhuang37/repos/VisQ-Search-Engine/examples', 'cell_type_stats_box.csv')
# df.to_csv(output_csv, index=False)

# for marker, file_name in glial_panel.items():
#     tif_path = os.path.join(file_path, file_name)
#     df = process_175boxes(df, tif_path, marker=marker)  
# output_csv = os.path.join('/home/lhuang37/repos/VisQ-Search-Engine/examples', 'glial_stats_box.csv')
# df.to_csv(output_csv, index=False)

# for marker, file_name in neuronal_connectivity_panel.items():
#     tif_path = os.path.join(file_path, file_name)
#     df = process_175boxes(df, tif_path, marker=marker)
# output_csv = os.path.join('/home/lhuang37/repos/VisQ-Search-Engine/examples', 'neuronal_connectivity_stats_box.csv')
# df.to_csv(output_csv, index=False)
file_name = 'S1_R2C8.tif'
tif_path = os.path.join(file_path,file_name)
output_csv = os.path.join('/home/lhuang37/repos/VisQ-Search-Engine/examples', 'tomato.csv')
df = process_boxes(df,tif_path,marker = 'Tomato Lectin')
df.to_csv(output_csv,index=False)