"""
Kaggle Data Pipeline

This module provides utilities for downloading from and uploading to Kaggle datasets.
"""

import os
import json
import pandas as pd
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleDataPipeline:
    """
    Handle downloading and uploading datasets to/from Kaggle.
    """
    
    def __init__(self, kaggle_json_path=None):
        """
        Initialize the Kaggle data pipeline.
        
        Parameters:
        kaggle_json_path (str or Path, optional): Path to kaggle.json credentials file
        """
        self.api = None
        self.kaggle_json_path = kaggle_json_path
        self._setup_credentials()
    
    def _setup_credentials(self):
        """Setup Kaggle API credentials"""
        if self.kaggle_json_path:
            kaggle_json = Path(self.kaggle_json_path)
            if kaggle_json.exists():
                os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_json.parent)
                print(f"✓ Using Kaggle config from: {kaggle_json.parent}")
            else:
                print(f"✗ Kaggle JSON not found at: {kaggle_json}")
                raise FileNotFoundError(f"kaggle.json not found at {kaggle_json}")
        
        # Authenticate
        try:
            self.api = KaggleApi()
            self.api.authenticate()
            print("✓ Authenticated with Kaggle API")
        except Exception as e:
            print(f"✗ Kaggle authentication failed: {e}")
            raise
    
    def download_dataset(self, dataset_name, download_path, file_name=None):
        """
        Download a dataset from Kaggle.
        
        Parameters:
        dataset_name (str): Dataset identifier (e.g., 'username/dataset-name')
        download_path (str or Path): Local directory to download to
        file_name (str, optional): Specific file to download (downloads all if None)
        
        Returns:
        Path: Path to downloaded file or directory
        """
        download_path = Path(download_path)
        download_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDownloading dataset: {dataset_name}")
        print(f"Destination: {download_path}")
        
        try:
            if file_name:
                # Download specific file
                self.api.dataset_download_file(
                    dataset=dataset_name,
                    file_name=file_name,
                    path=download_path
                )
                
                # Unzip if needed
                zip_file = download_path / f"{file_name}.zip"
                if zip_file.exists():
                    import zipfile
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(download_path)
                    zip_file.unlink()  # Remove zip file
                    
                file_path = download_path / file_name
            else:
                # Download all files
                self.api.dataset_download_files(
                    dataset=dataset_name,
                    path=download_path,
                    unzip=True,
                    quiet=False
                )
                file_path = download_path
            
            print(f"✓ Dataset downloaded successfully")
            return file_path
        
        except Exception as e:
            print(f"✗ Download failed: {e}")
            raise
    
    def load_dataset(self, file_path):
        """
        Load a dataset file into a pandas DataFrame.
        
        Parameters:
        file_path (str or Path): Path to the dataset file
        
        Returns:
        pd.DataFrame: Loaded dataset
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"\nLoading: {file_path.name}")
        
        # Determine file type and load
        suffix = file_path.suffix.lower()
        
        if suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif suffix == '.csv':
            df = pd.read_csv(file_path)
        elif suffix == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        print(f"✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    
    def save_dataset(self, df, file_path, file_format='parquet'):
        """
        Save a DataFrame to disk.
        
        Parameters:
        df (pd.DataFrame): DataFrame to save
        file_path (str or Path): Output file path
        file_format (str): Format ('parquet', 'csv', 'json')
        
        Returns:
        Path: Path to saved file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving: {file_path.name}")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        if file_format == 'parquet':
            df.to_parquet(file_path, index=False, compression='snappy')
        elif file_format == 'csv':
            df.to_csv(file_path, index=False)
        elif file_format == 'json':
            df.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved: {file_size_mb:.2f} MB")
        
        return file_path
    
    def create_dataset_metadata(self, dataset_slug, title, subtitle="", description="", 
                                 is_public=True, licenses_name="CC0-1.0"):
        """
        Create metadata JSON for a new Kaggle dataset.
        
        Parameters:
        dataset_slug (str): URL-friendly dataset identifier
        title (str): Dataset title
        subtitle (str): Short subtitle
        description (str): Full description
        is_public (bool): Whether dataset is public
        licenses_name (str): License type
        
        Returns:
        dict: Metadata dictionary
        """
        return {
            "title": title,
            "id": dataset_slug,
            "subtitle": subtitle,
            "description": description,
            "isPrivate": not is_public,
            "licenses": [{"name": licenses_name}],
            "keywords": [],
            "collaborators": [],
            "data": []
        }
    
    def upload_dataset(self, dataset_dir, dataset_slug, title, subtitle="", description="",
                       is_public=True, version_notes="Updated dataset"):
        """
        Upload or update a dataset to Kaggle.
        
        Parameters:
        dataset_dir (str or Path): Directory containing dataset files
        dataset_slug (str): Dataset slug (e.g., 'username/dataset-name')
        title (str): Dataset title
        subtitle (str): Short subtitle
        description (str): Full description
        is_public (bool): Whether dataset should be public
        version_notes (str): Notes for this version
        
        Returns:
        bool: True if successful
        """
        dataset_dir = Path(dataset_dir)
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        print(f"\nPreparing to upload dataset: {dataset_slug}")
        print(f"  Directory: {dataset_dir}")
        
        # Create metadata file
        metadata = self.create_dataset_metadata(
            dataset_slug=dataset_slug,
            title=title,
            subtitle=subtitle,
            description=description,
            is_public=is_public
        )
        
        metadata_path = dataset_dir / 'dataset-metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Created metadata file")
        
        # Check if dataset already exists
        try:
            # Try to get existing dataset
            self.api.dataset_metadata(dataset_slug, dataset_dir)
            
            # Dataset exists, create new version
            print(f"  Dataset exists, creating new version...")
            self.api.dataset_create_version(
                folder=str(dataset_dir),
                version_notes=version_notes,
                quiet=False,
                convert_to_csv=False,
                delete_old_versions=False
            )
            print(f"✓ New version created successfully")
            
        except Exception as e:
            # Dataset doesn't exist, create new
            print(f"  Dataset doesn't exist, creating new...")
            try:
                self.api.dataset_create_new(
                    folder=str(dataset_dir),
                    public=is_public,
                    quiet=False,
                    convert_to_csv=False,
                    dir_mode='tar'
                )
                print(f"✓ Dataset created successfully")
            except Exception as create_error:
                print(f"✗ Upload failed: {create_error}")
                raise
        
        return True
    
    def dataset_exists(self, dataset_slug):
        """
        Check if a dataset exists on Kaggle.
        
        Parameters:
        dataset_slug (str): Dataset slug (e.g., 'username/dataset-name')
        
        Returns:
        bool: True if dataset exists
        """
        try:
            self.api.dataset_status(dataset_slug)
            return True
        except:
            return False
