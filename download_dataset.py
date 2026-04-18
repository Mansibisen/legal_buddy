#!/usr/bin/env python3
"""
Dataset Downloader for Multi-Modal Legal System

Flexible script to download legal datasets:
- Pile-of-Law
- Other legal document datasets (configurable)

Usage:
    python download_dataset.py                    # Interactive mode
    python download_dataset.py --dataset pile-of-law
    python download_dataset.py --dataset custom --source https://example.com/data.zip
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request
import urllib.error
import zipfile
import tarfile
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetConfig:
    """Configuration for different datasets"""
    
    DATASETS = {
        "pile-of-law": {
            "name": "Pile-of-Law Dataset",
            "description": "Comprehensive legal document dataset",
            "source": "https://the-eye.eu/public/AI/pile_v2/pile-of-law/",
            "files": [
                "pile_of_law_docs.jsonl.zst",  # Main dataset
            ],
            "type": "jsonl",  # Format: jsonl, pdf, txt, csv
            "size_gb": 5.0,  # Approximate size
            "sample_files": [
                "https://huggingface.co/datasets/pile-of-law/pile-of-law/raw/main/data/pile_of_law_sample.jsonl"
            ]
        },
        "legal-case-reports": {
            "name": "Legal Case Reports",
            "description": "Curated legal case reports and judgments",
            "source": "https://example.com/legal-cases/",
            "files": ["cases.tar.gz"],
            "type": "txt",
            "size_gb": 2.0
        },
        "contracts": {
            "name": "Contract Corpus",
            "description": "Sample contracts and agreements",
            "source": "https://example.com/contracts/",
            "files": ["contracts.zip"],
            "type": "pdf",
            "size_gb": 1.5
        }
    }
    
    @classmethod
    def get_dataset(cls, name: str) -> Optional[Dict]:
        """Get dataset configuration by name"""
        return cls.DATASETS.get(name.lower())
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all available datasets"""
        return list(cls.DATASETS.keys())


class DatasetDownloader:
    """Download and process datasets"""
    
    def __init__(self, output_dir: str = "./data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir.parent / "dataset_metadata.json"
    
    def print_header(self, text: str):
        """Print formatted header"""
        width = 70
        print("\n" + "=" * width)
        print(text.center(width))
        print("=" * width)
    
    def print_success(self, text: str):
        """Print success message"""
        print(f"\n✅ {text}")
    
    def print_error(self, text: str):
        """Print error message"""
        print(f"\n❌ {text}")
    
    def print_info(self, text: str):
        """Print info message"""
        print(f"📝 {text}")
    
    def print_warning(self, text: str):
        """Print warning message"""
        print(f"⚠️  {text}")
    
    def download_file(self, url: str, destination: Path, description: str = "") -> bool:
        """Download a file with progress"""
        try:
            self.print_info(f"Downloading: {description or url}")
            print(f"   Source: {url}")
            print(f"   Destination: {destination}")
            
            # Simple progress bar
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size)
                print(f"   Progress: {percent}% ", end='\r')
            
            urllib.request.urlretrieve(url, destination, download_progress)
            print()  # New line after progress
            self.print_success(f"Downloaded: {destination.name}")
            return True
            
        except urllib.error.URLError as e:
            self.print_error(f"Failed to download {url}: {e}")
            return False
        except Exception as e:
            self.print_error(f"Unexpected error downloading {url}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path = None) -> bool:
        """Extract zip or tar.gz files"""
        try:
            extract_to = extract_to or self.output_dir
            extract_to.mkdir(parents=True, exist_ok=True)
            
            self.print_info(f"Extracting: {archive_path.name}")
            
            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffixes[-2:] == ['.tar', '.gz']:
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                self.print_error(f"Unsupported archive format: {archive_path}")
                return False
            
            self.print_success(f"Extracted to: {extract_to}")
            return True
            
        except Exception as e:
            self.print_error(f"Failed to extract {archive_path}: {e}")
            return False
    
    def convert_jsonl_to_pdf(self, jsonl_file: Path) -> bool:
        """Convert JSONL dataset to PDF-like documents"""
        try:
            self.print_info(f"Converting JSONL to documents: {jsonl_file.name}")
            
            import json
            
            txt_dir = self.output_dir / "converted_documents"
            txt_dir.mkdir(parents=True, exist_ok=True)
            
            count = 0
            with open(jsonl_file, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        doc = json.loads(line)
                        
                        # Extract text content
                        text = doc.get('text', '')
                        title = doc.get('title', f'Document_{line_num}')
                        
                        if text:
                            # Save as .txt file
                            filename = f"{count:06d}_{title[:50].replace('/', '_')}.txt"
                            filepath = txt_dir / filename
                            
                            with open(filepath, 'w') as out:
                                out.write(f"Title: {title}\n")
                                out.write(f"Source: {doc.get('source', 'Unknown')}\n")
                                out.write("=" * 80 + "\n\n")
                                out.write(text)
                            
                            count += 1
                            
                            if count % 100 == 0:
                                print(f"   Converted {count} documents...", end='\r')
                    
                    except json.JSONDecodeError:
                        continue
            
            print()  # New line
            self.print_success(f"Converted {count} documents to text files")
            self.print_info(f"Documents saved to: {txt_dir}")
            
            return count > 0
            
        except Exception as e:
            self.print_error(f"Failed to convert JSONL: {e}")
            return False
    
    def save_metadata(self, dataset_name: str, config: Dict):
        """Save dataset metadata"""
        try:
            metadata = {
                "dataset_name": dataset_name,
                "config": config,
                "downloaded_at": str(Path.cwd()),
                "output_directory": str(self.output_dir),
                "files": [f.name for f in self.output_dir.glob("*")]
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.print_info(f"Metadata saved to: {self.metadata_file}")
        except Exception as e:
            self.print_warning(f"Failed to save metadata: {e}")
    
    def download_dataset(self, dataset_name: str, url: Optional[str] = None) -> bool:
        """Download a dataset"""
        self.print_header(f"DOWNLOADING DATASET: {dataset_name.upper()}")
        
        # Get dataset config
        config = DatasetConfig.get_dataset(dataset_name)
        
        if not config:
            self.print_error(f"Unknown dataset: {dataset_name}")
            self.print_info("Available datasets:")
            for ds in DatasetConfig.list_datasets():
                self.print_info(f"  • {ds}")
            return False
        
        # Use custom URL if provided
        if url:
            config = config.copy()
            config['source'] = url
        
        self.print_info(f"Dataset: {config['name']}")
        self.print_info(f"Description: {config['description']}")
        self.print_info(f"Type: {config['type']}")
        self.print_info(f"Approximate size: {config['size_gb']} GB")
        
        # Confirm download
        response = input("\n🔹 Proceed with download? (y/n): ").strip().lower()
        if response != 'y':
            self.print_info("Download cancelled")
            return False
        
        # Download files
        downloaded_files = []
        
        if 'sample_files' in config:
            # Download sample files for testing
            self.print_info("Using sample dataset for testing...")
            for url in config.get('sample_files', []):
                filename = Path(url).name
                dest = self.output_dir / filename
                if self.download_file(url, dest, f"Sample: {filename}"):
                    downloaded_files.append(dest)
        else:
            # Download from main source
            for file in config.get('files', []):
                url = config['source'].rstrip('/') + '/' + file
                dest = self.output_dir / file
                if self.download_file(url, dest, file):
                    downloaded_files.append(dest)
        
        if not downloaded_files:
            self.print_error("No files downloaded successfully")
            return False
        
        # Extract archives
        for file_path in downloaded_files:
            if file_path.suffix in ['.zip'] or str(file_path).endswith('.tar.gz'):
                if not self.extract_archive(file_path):
                    self.print_warning(f"Failed to extract {file_path.name}")
        
        # Convert JSONL if needed
        if config['type'] == 'jsonl':
            jsonl_files = list(self.output_dir.glob("*.jsonl"))
            if jsonl_files:
                for jsonl_file in jsonl_files:
                    self.convert_jsonl_to_pdf(jsonl_file)
        
        # Save metadata
        self.save_metadata(dataset_name, config)
        
        # Print summary
        self.print_success("Dataset downloaded successfully!")
        self.print_info(f"Location: {self.output_dir}")
        self.print_info(f"Total files: {len(list(self.output_dir.glob('*')))}")
        
        return True


def interactive_mode():
    """Interactive dataset selection"""
    print("\n" + "=" * 70)
    print("DATASET DOWNLOADER - INTERACTIVE MODE".center(70))
    print("=" * 70)
    
    # List available datasets
    print("\n📚 AVAILABLE DATASETS:")
    datasets = DatasetConfig.list_datasets()
    for i, ds in enumerate(datasets, 1):
        config = DatasetConfig.get_dataset(ds)
        print(f"\n   {i}. {config['name']}")
        print(f"      • {config['description']}")
        print(f"      • Size: ~{config['size_gb']} GB")
    
    # Select dataset
    choice = input("\n🔹 Select dataset (number or name, or 'q' to quit): ").strip()
    
    if choice.lower() == 'q':
        print("Cancelled")
        return
    
    # Parse choice
    if choice.isdigit() and 1 <= int(choice) <= len(datasets):
        dataset_name = datasets[int(choice) - 1]
    else:
        dataset_name = choice
    
    # Output directory
    output_dir = input("🔹 Output directory (default: ./data/raw): ").strip() or "./data/raw"
    
    # Download
    downloader = DatasetDownloader(output_dir)
    downloader.download_dataset(dataset_name)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Download legal datasets for the Multi-Modal Legal System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_dataset.py                              # Interactive mode
  python download_dataset.py --dataset pile-of-law        # Download Pile-of-Law
  python download_dataset.py --list                       # List datasets
  python download_dataset.py --dataset custom \\
    --source https://example.com/data.zip \\
    --output ./data/custom                                # Custom source
        """
    )
    
    parser.add_argument(
        '--dataset',
        help='Dataset to download (pile-of-law, contracts, legal-case-reports, etc.)'
    )
    parser.add_argument(
        '--source',
        help='Custom URL source for dataset'
    )
    parser.add_argument(
        '--output',
        default='./data/raw',
        help='Output directory (default: ./data/raw)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available datasets'
    )
    
    args = parser.parse_args()
    
    # List datasets
    if args.list:
        print("\n📚 AVAILABLE DATASETS:")
        for ds in DatasetConfig.list_datasets():
            config = DatasetConfig.get_dataset(ds)
            print(f"\n  {ds.upper()}")
            print(f"    Name: {config['name']}")
            print(f"    Description: {config['description']}")
            print(f"    Type: {config['type']}")
            print(f"    Size: ~{config['size_gb']} GB")
        return
    
    # Interactive mode if no dataset specified
    if not args.dataset:
        interactive_mode()
        return
    
    # Download specified dataset
    downloader = DatasetDownloader(args.output)
    downloader.download_dataset(args.dataset, args.source)


if __name__ == "__main__":
    main()
