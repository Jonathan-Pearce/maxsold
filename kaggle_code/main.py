
import kaggle
from pathlib import Path
from read_data import download_kaggle_dataset, read_dataset_file, search_kaggle_datasets, list_dataset_files

# Example 1: Download and read a specific dataset
dataset_name = "pearcej/raw-maxsold-item-enriched"
dataset_dir = download_kaggle_dataset(dataset_name, './data')
df = read_dataset_file(dataset_dir / 'file.csv')

# Example 2: Search for datasets first
#datasets = search_kaggle_datasets("titanic", max_results=5)
# Then download the one you want
#download_kaggle_dataset(datasets[0].ref, './data')

# Example 3: Download and read in one go
#download_path = download_kaggle_dataset("datasnaek/chess", './data')
#files = list_dataset_files(download_path)
#df = read_dataset_file(files[0])