import numpy as np
import tifffile
from ome_types import OME, model
from scipy.ndimage import zoom
import os

def generate_pyramid(base_image, num_levels, axes, dtype=None):
    """Generate pyramid levels with exact level count matching reference"""
    pyramid = [base_image]
    current = base_image
    if dtype is None:
        for _ in range(num_levels - 1):
            zoom_factors = [0.5 if ax in 'YX' else 1 for ax in axes]
            downsampled = zoom(current, zoom_factors, order=1)
            pyramid.append(downsampled)
            del current
            current = downsampled
    else:
        for _ in range(num_levels - 1):
            zoom_factors = [0.5 if ax in 'YX' else 1 for ax in axes]
            downsampled = zoom(current, zoom_factors, order=1).astype(dtype)
            pyramid.append(downsampled)
            del current
            current = downsampled
    
    return pyramid # Strictly enforce level count


#**********************************************************************************
def construct(output_file, asf_path, ref_levels, ref_axes, dimention_order, datatype,reference_pixel_size,marker_dict):
    biomarkers = list(marker_dict.keys())
    print(biomarkers)
    # Load and stack channels,
    processed_channels = []
    for biomarker in biomarkers:
        print('Checking:', biomarker)
        # channel_path = os.path.join(asf_path, f"S1_{channel_dict[biomarker]}.tif")
        # channel_path = os.path.join(asf_path, f"{cell_type_panel[biomarker]}")
        file_name = marker_dict[biomarker]
        channel_path = os.path.join(asf_path, file_name)
        processed_channels.append(tifffile.imread(channel_path).astype(datatype))
        

    c_axis = ref_axes.index('C')
    processed_base = np.stack(processed_channels, axis=c_axis)

    # Generate pyramid with EXACTLY ref_levels levels
    processed_pyramid = generate_pyramid(processed_base, ref_levels, ref_axes, datatype)
    print('pyramid shape:', [level.shape for level in processed_pyramid])

    # Create OME metadata
    new_ome = OME()
    ome_channels = [model.Channel(id=f"Channel:0:{i}", name=name,samples_per_pixel=1) 
                for i, name in enumerate(biomarkers)]

    x_dim = ref_axes.index('X')
    y_dim = ref_axes.index('Y')
    size_x = processed_base.shape[x_dim]
    size_y = processed_base.shape[y_dim]

    new_pixels = model.Pixels(
        dimension_order=dimention_order,
        type=processed_base.dtype.name,
        size_c=len(biomarkers),
        size_z=1,
        size_t=1,
        size_x=size_x,
        size_y=size_y,
        channels=ome_channels,
        physical_size_x=reference_pixel_size,
        physical_size_y=reference_pixel_size,
    )
    del processed_base
    new_ome.images.append(model.Image(
        id="Image:0",
        name="Generated Image",
        pixels=new_pixels
    ))

    # Write with strict pyramid validation
    with tifffile.TiffWriter(output_file, bigtiff=True) as tif:
        # Write base level
        tif.write(
            processed_pyramid[0],
            photometric='minisblack',
            description=new_ome.to_xml(),
            compression='zlib',
            subifds=len(processed_pyramid)-1,

        )
        
        # Write pyramid levels
        print('total levels:', len(processed_pyramid))
        for i, level in enumerate(processed_pyramid[1:]):
            if level.size == 0:  # Skip empty placeholder levels
                continue
            print(f"Processing level {i+1} ")
            tif.write(
                level,
                photometric='minisblack',
                subfiletype=1,
                compression='zlib',
            )
            processed_pyramid[i+1] = None
            

# User inputs
PIXEL_SIZE_NM = 330  # Pixel size in nanometers
reference_pixel_size = PIXEL_SIZE_NM / 1000  # Convert nm to µm
datatype = np.uint16 
 
marker_dict =  {
    "R1C2.tif": "DAPI",
    "R1C10.tif": "NFH",
    "R1C3.tif": "CC3",
    "R1C4.tif": "NeuN",
    "R1C5.tif": "MBP",
    "R1C6.tif": "RECA1",
    "R1C7.tif": "IBA1++",  
    "R1C8.tif": "TomatoLectin",
    "R1C9.tif": "PCNA",
    "R2C10.tif": "MAP2",
    "R2C3.tif": "GAD67",
    "R2C4.tif": "GFAP",
    "R2C5.tif": "Parvalbumin",
    "R2C6.tif": "S100",
    "R2C7.tif": "Calretinin",
    "R2C8.tif": "TomatoLectin",
    "R2C9.tif": "CD31"
}
# revert the marker_dict
marker_dict = {v: k for k, v in marker_dict.items()}
ref_levels = 8
ref_axes = 'CYX'
dimention_order = 'XYCZT'
datatype = 'uint16'
pixel_size = 330
# run a iteration of a list of folders
mother_path = '/home/lhuang37/datasets/TBI'
files_1 =[f for f in os.listdir(mother_path) 
           if os.path.isdir(os.path.join(mother_path, f))]
for folder in files_1:
    parent_folder = os.path.join(mother_path, folder)
    # list all folders and ignore files
    files_2 = [f for f in os.listdir(parent_folder) 
           if os.path.isdir(os.path.join(parent_folder, f))]
    for subfolder in files_2:
       
        if os.path.isdir(os.path.join(mother_path, folder, subfolder)):
          
            output = os.path.join(mother_path, folder, subfolder,'whole_channels.ome.tif' )
            asf_path = os.path.join(mother_path, folder, subfolder,'final')

            print('Processing folder:', os.path.join(mother_path, folder, subfolder))
                
                
            construct(output, asf_path, ref_levels, ref_axes, dimention_order, datatype,reference_pixel_size,marker_dict)
