#!/usr/bin/env python3
"""
Upload or update a dataset to Kaggle.

Usage:
    python upload_to_kaggle.py
"""

import os
import json
import shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def setup_kaggle_config():
    """Set up Kaggle configuration to use custom kaggle.json location"""
    custom_kaggle_json = Path('/workspaces/maxsold/.kaggle/kaggle.json')
    default_kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    # If custom location exists and default doesn't, set environment variable
    if custom_kaggle_json.exists() and not default_kaggle_json.exists():
        os.environ['KAGGLE_CONFIG_DIR'] = str(custom_kaggle_json.parent)
        print(f"✓ Using Kaggle config from: {custom_kaggle_json.parent}")
        return custom_kaggle_json
    elif default_kaggle_json.exists():
        print(f"✓ Using Kaggle config from: {default_kaggle_json.parent}")
        return default_kaggle_json
    else:
        print(f"✗ kaggle.json not found in either location:")
        print(f"  - {custom_kaggle_json}")
        print(f"  - {default_kaggle_json}")
        return None


def check_kaggle_auth():
    """Check if Kaggle API is properly authenticated"""
    print("Checking Kaggle authentication...")
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        # Try to verify credentials by listing competitions (requires valid auth)
        try:
            api.competitions_list(page=1, page_size=1)
            print("✓ Successfully authenticated with Kaggle API")
            print(f"✓ Using credentials from: {api.config_file}")
            return api
        except Exception as e:
            print(f"✗ Authentication succeeded but API calls failing: {e}")
            print("\nThis usually means:")
            print("  1. Your API token is expired - regenerate it at https://www.kaggle.com/settings/account")
            print("  2. Your token doesn't have write permissions")
            print("  3. Network/firewall issues")
            return None
            
    except Exception as e:
        print(f"✗ Kaggle authentication failed: {e}")
        print("\nTo fix this:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New Token' under API section")
        print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return None


def verify_credentials():
    """Verify Kaggle credentials have proper permissions"""
    print("\nVerifying Kaggle credentials...")
    
    # Check both possible locations
    kaggle_json_locations = [
        Path('/workspaces/maxsold/.kaggle/kaggle.json'),
        Path.home() / '.kaggle' / 'kaggle.json',
        Path.home() / '.config' / 'kaggle' / 'kaggle.json'
    ]
    
    kaggle_json = None
    for location in kaggle_json_locations:
        if location.exists():
            kaggle_json = location
            break
    
    if not kaggle_json:
        print(f"✗ kaggle.json not found in any of these locations:")
        for loc in kaggle_json_locations:
            print(f"  - {loc}")
        return False
    
    print(f"✓ Found kaggle.json at: {kaggle_json}")
    
    # Check file permissions
    import stat
    mode = oct(os.stat(kaggle_json).st_mode)[-3:]
    if mode != '600':
        print(f"⚠ Warning: kaggle.json has permissions {mode}, should be 600")
        print(f"  Run: chmod 600 {kaggle_json}")
    
    # Read and display username (not the key)
    try:
        with open(kaggle_json) as f:
            creds = json.load(f)
            username = creds.get('username', 'unknown')
            has_key = 'key' in creds
            print(f"✓ Found credentials for user: {username}")
            print(f"✓ API key present: {has_key}")
            
            if not username or not has_key:
                print("✗ Invalid kaggle.json format")
                return False
            
            return True
    except Exception as e:
        print(f"✗ Error reading kaggle.json: {e}")
        return False


def get_kaggle_username():
    """Get Kaggle username from kaggle.json"""
    kaggle_json_locations = [
        Path('/workspaces/maxsold/.kaggle/kaggle.json'),
        Path.home() / '.kaggle' / 'kaggle.json',
        Path.home() / '.config' / 'kaggle' / 'kaggle.json'
    ]
    
    for location in kaggle_json_locations:
        if location.exists():
            try:
                with open(location) as f:
                    return json.load(f)['username']
            except:
                pass
    
    return 'username'


def create_dataset_metadata(dataset_dir, dataset_slug, title, description=""):
    """
    Create dataset-metadata.json file required for Kaggle upload.
    
    Args:
        dataset_dir: Directory containing the dataset files
        dataset_slug: Short name for the dataset (e.g., 'raw-maxsold-item')
        title: Display title for the dataset
        description: Description of the dataset
    """
    dataset_dir = Path(dataset_dir)
    
    # Get list of files in the directory
    files = [f.name for f in dataset_dir.iterdir() if f.is_file() and f.name != 'dataset-metadata.json']
    
    username = get_kaggle_username()
    
    metadata = {
        "title": title,
        "id": f"{username}/{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],  # Public domain license
        "description": description,
        "keywords": ["maxsold", "auction", "items", "ecommerce"],
        "resources": [
            {
                "path": filename,
                "description": f"Dataset file: {filename}"
            }
            for filename in files
        ]
    }
    
    metadata_path = dataset_dir / 'dataset-metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created metadata file: {metadata_path}")
    print(f"  Dataset ID: {metadata['id']}")
    return metadata_path


def upload_dataset_file(source_file, dataset_slug, version_notes="Updated dataset"):
    """
    Upload a file to an existing Kaggle dataset.
    
    Args:
        source_file: Path to the file to upload
        dataset_slug: Dataset identifier (e.g., 'pearcej/raw-maxsold-item')
        version_notes: Notes for this version
    
    Returns:
        bool: True if successful
    """
    source_file = Path(source_file)
    
    if not source_file.exists():
        print(f"✗ Source file not found: {source_file}")
        return False
    
    # Verify credentials first
    if not verify_credentials():
        print("\n⚠ Please fix credential issues before uploading")
        return False
    
    # Get API
    api = check_kaggle_auth()
    if not api:
        return False
    
    # Create a temporary directory for the upload
    temp_dir = Path('./temp_kaggle_upload')
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Copy file to temp directory
        dest_file = temp_dir / source_file.name
        print(f"\nCopying file to temp directory...")
        print(f"  Source: {source_file}")
        print(f"  Size: {source_file.stat().st_size / (1024*1024):.2f} MB")
        
        shutil.copy2(source_file, dest_file)
        
        # Parse dataset slug - remove username if provided since we'll get it from credentials
        if '/' in dataset_slug:
            _, dataset_name = dataset_slug.split('/')
        else:
            dataset_name = dataset_slug
        
        # Create metadata
        title = dataset_name.replace('-', ' ').title()
        description = f"MaxSold dataset containing {source_file.name}. Updated on 2025-12-23."
        
        create_dataset_metadata(
            temp_dir,
            dataset_name,
            title,
            description
        )
        
        # Get username from credentials
        username = get_kaggle_username()
        
        # Check if dataset exists
        dataset_exists = True
        try:
            api.dataset_view(f"{username}/{dataset_name}")
            dataset_exists = True
            print(f"\n✓ Dataset exists: {username}/{dataset_name}")
        except Exception:
            print(f"\n⚠ Dataset does not exist yet: {username}/{dataset_name}")
            print("  Will attempt to create new dataset...")
        
        # Upload or create dataset
        if dataset_exists:
            print(f"\nUploading new version to existing dataset...")
            print("This may take a few minutes...")
            api.dataset_create_version(
                folder=str(temp_dir),
                version_notes=version_notes,
                quiet=False,
                delete_old_versions=False
            )
            print(f"✓ Successfully uploaded new version!")
        else:
            print(f"\nCreating new dataset...")
            print("This may take a few minutes...")
            print("\nNote: If you get 401 Unauthorized, your API token may need to be regenerated.")
            print("Go to https://www.kaggle.com/settings/account and create a new API token.\n")
            
            api.dataset_create_new(
                folder=str(temp_dir),
                public=True,
                quiet=False
            )
            print(f"✓ Successfully created new dataset!")
        
        print(f"\nDataset URL: https://www.kaggle.com/datasets/{username}/{dataset_name}")
        print("\n⚠ Note: It may take a few minutes for the dataset to appear on Kaggle's website.")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ Error uploading dataset: {e}")
        
        if '401' in error_msg or 'Unauthorized' in error_msg:
            print("\n" + "="*80)
            print("AUTHENTICATION ERROR - YOUR API TOKEN NEEDS TO BE REGENERATED")
            print("="*80)
            print("\nSteps to fix:")
            print("1. Go to https://www.kaggle.com/settings/account")
            print("2. Scroll down to the 'API' section")
            print("3. Click 'Expire API Token' if one exists")
            print("4. Click 'Create New API Token'")
            print("5. Download the new kaggle.json file")
            print("6. Move it to ~/.kaggle/kaggle.json")
            print("7. Run: chmod 600 ~/.kaggle/kaggle.json")
            print("8. Try uploading again")
        elif '403' in error_msg or 'Forbidden' in error_msg:
            print("\nPossible issues:")
            print("  - Dataset slug might be taken by another user")
            print("  - You don't have permission to upload datasets")
            print("  - Phone verification required on Kaggle account")
        
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n✓ Cleaned up temp directory")


def main():
    """Upload items_details dataset to Kaggle"""
    
    print("="*80)
    print("UPLOAD DATASET TO KAGGLE")
    print("="*80)
    
    # Set up Kaggle config directory
    config_file = setup_kaggle_config()
    if not config_file:
        print("\n✗ Cannot proceed without kaggle.json")
        print("\nQuick fix:")
        print("  mkdir -p ~/.kaggle")
        print("  cp /workspaces/maxsold/.kaggle/kaggle.json ~/.kaggle/")
        print("  chmod 600 ~/.kaggle/kaggle.json")
        return
    
    # Configuration
    source_file = "/workspaces/maxsold/data/raw_data/items_details/items_details_20251222.parquet"
    dataset_slug = "raw-maxsold-item"  # Don't include username, it will be auto-detected
    version_notes = "Updated items_details dataset - 2025-12-22"
    
    # Verify file exists
    source_path = Path(source_file)
    if not source_path.exists():
        print(f"✗ Source file not found: {source_file}")
        print("\nAvailable files in directory:")
        parent_dir = source_path.parent
        if parent_dir.exists():
            for f in sorted(parent_dir.glob('*.parquet')):
                size_mb = f.stat().st_size / (1024*1024)
                print(f"  - {f.name} ({size_mb:.2f} MB)")
        return
    
    # Upload dataset
    success = upload_dataset_file(
        source_file=source_file,
        dataset_slug=dataset_slug,
        version_notes=version_notes
    )
    
    if success:
        print("\n" + "="*80)
        print("UPLOAD COMPLETE")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("UPLOAD FAILED - SEE INSTRUCTIONS ABOVE")
        print("="*80)


if __name__ == "__main__":
    main()