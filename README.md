A Visual Query-Driven Search Engine for Brain Tissue Image Analysis
This project provides a search engine designed for analyzing multiplex fluorescence images of brain tissue. It combines multiple image signals into a unified feature space and constructs a community map to highlight similar and distinct regions within the tissue.
Project Overview
The Visual Query-Driven Search Engine (VisQ-Search-Engine) enables researchers to process and analyze large-scale brain tissue images by:

Cropping images into smaller patches for efficient processing.
Training a neural network to create a feature space for querying.
Supporting interactive exploration through integration with QuPath.
Providing tools for profiling and post-query analysis.

This tutorial guides you through installation, data preparation, training, querying, profiling, and integration with QuPath.
Installation
Prerequisites
Ensure you have the following prerequisites installed:

CUDA Driver: Version 12.1
PyTorch: Version 2.3.0
Conda: For installing PyTorch and other dependencies
Python: Compatible with the required packages

Installing Dependencies

Install PyTorch with CUDA support using Conda:
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia


Install the required Python packages using pip:
pip install torch==2.3.0 numpy==1.26.4 torchvision==0.18.0 six==1.16.0 h5py==3.11.0 Pillow==9.4.0 scipy==1.12.0 timm==1.0.11 scikit-learn==1.4.2 metric-learn==0.6.2


Install additional packages that are only available via pip:
pip install faiss-gpu-cu12 infomap==2.8.0 scikit-image==0.22.0 tqdm==4.66.4 einops==0.8.0


Clone and install the repository as a package:
git clone https://github.com/RoysamLab/VisQ-Search-Engine.git
cd VisQ-Search-Engine
pip install -e .



Data Preparation
Before using the search engine, you must prepare your dataset by cropping large brain images into smaller patches of size [174, 174, n], where n is the number of channels. The prepared data is used for training and querying.
Directory Structure
Organize your project directory as follows:
VisQ-Search-Engine/
├── clustercontrast/
├── examples/
│   ├── data/
│   ├── logs/
│   ├── pretrained/
├── results/
├── runs/
├── bash.sh
├── setup.py

Preparing the Dataset
Run the following command to crop images into patches:
python examples/makeDS.py \
  --INPUT_DIR <path_to_single_channel_images> \
  --OUTPUT_DIR <output_directory_for_patches> \
  --BBXS_FILE <csv_file_with_cell_detections> \
  --BIO_MARKERS_PANNEL <panel_name_key>


INPUT_DIR: Path to the folder containing single-channel brain images.
OUTPUT_DIR: Path to save the cropped patches (e.g., [174, 174, 10]).
BBXS_FILE: CSV file containing cell detection data with columns: centroid_x, centroid_y, x_min, x_max, y_min, y_max.
BIO_MARKERS_PANNEL: Key for the biomarker panel defined in examples/channel_lists.

Note: Default parameters can be modified in examples/makeDS.py if needed.
Training the Model
Train the neural network to create the feature space for querying. The following command trains the model with a UNet backbone:
python examples/graph_train.py \
  -b 256 \
  -a unet \
  -d brain \
  -dn <dataset_folder_name> \
  -nc 5 \
  --iters 200 \
  --momentum 0.2 \
  --eps 0.6 \
  --num-instances 16 \
  --height 50 \
  --width 50 \
  --epochs 50 \
  --logs-dir examples/logs/<log_folder_name>/

Arguments

-b: Batch size (e.g., 256).
-a: Backbone network (e.g., unet).
-d: Dataset type (e.g., brain).
-dn: Dataset folder name.
-nc: Number of channels (e.g., 5).
--iters, --momentum, --eps, --num-instances: Training hyperparameters.
--height, --width: Patch dimensions (e.g., 50x50).
--epochs: Number of training epochs.
--logs-dir: Directory to save training logs.

Note: Adjust hyperparameters as needed, but the provided defaults are recommended for initial training.
Querying the Search Engine
After training, use the search engine to query the dataset. The process involves two steps: running the query and post-processing the results.
Step 1: Run the Query
python examples/RWM_testUnet.py \
  -d brain \
  -a unet \
  --resume <path_to_model_checkpoint/model_best.pth.tar> \
  --height 50 \
  --width 50 \
  -nc 5 \
  -b 4096 \
  --data-dir <dataset_path> \
  -dn <dataset_folder_name> \
  --output-tag <output_file_tag>


--resume: Path to the trained model checkpoint.
--data-dir: Path to the prepared dataset.
--output-tag: Tag for the output file name.

Step 2: Post-Query Processing
Combine features from multiple panels (if applicable) and perform post-query analysis:
python examples/info_prequery.py \
  --feature_path <log_folder_path_panel1/bestraw_features.pkl> \
  --feature_path2 <log_folder_path_panel2/bestraw_features.pkl> \
  --feature_path3 <log_folder_path_panel3/bestraw_features.pkl> \
  --saved_path <output_folder_path/bestraw_features.pkl> \
  --mode '3in1'

Then, finalize the query:
python examples/post_query.py \
  --door <output_file_from_previous_step>

Profiling
Profile the dataset to analyze the results:
python examples/profileing.py \
  --dataset <panel_name> \
  -b <batch_size> \
  --outputCSV <output_file_name>.csv \
  --panel <panel_name>


--dataset: Name of the dataset panel.
-b: Batch size (recommended: 4096 or larger).
--outputCSV: Name of the output CSV file.
--panel: Panel name for profiling.

Integration with QuPath
To interactively explore the search engine results, integrate with QuPath:

Download and install QuPath 0.5.0.
Copy the .jar files from the Cell Search Engine repository to the QuPath extensions directory:
In QuPath, go to Extensions → Extension Manager → Open extension directory.
Place the .jar files in the opened directory.


Restart QuPath.
Access the search engine via Extensions → Cell Search Engine.

For detailed instructions, refer to the Cell Search Engine GitHub repository.
Acknowledgments
This work was supported by the National Institutes of Health (NINDS) under grant R01NS109118.
Citation
If you use this codebase in your work, please cite:
@article{roysam2025,
    author = {Roysam Lab},
    title = {A Visual Query-Driven Search Engine for Brain Tissue Image Analysis},
    journal = {TBD},
    year = {2025},
    doi = {TBD}
}

License
This project is licensed under the terms specified in the repository. Please review the license file for details.
