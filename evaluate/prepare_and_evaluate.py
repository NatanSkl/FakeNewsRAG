import os
import sys
import time
import subprocess
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))


def check_data_downloaded():
    print("="*80)
    print("STEP 1: CHECKING DOWNLOADED DATA")
    print("="*80)
    
    data_dir = Path("index/data")
    required_parts = ["news.csv.zip"] + [f"news.csv.z{i:02d}" for i in range(1, 10)]
    
    missing = []
    existing = []
    
    for part in required_parts:
        path = data_dir / part
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            existing.append(f"{part} ({size_mb:.1f} MB)")
        else:
            missing.append(part)
    
    print(f"\nExisting parts: {len(existing)}/10")
    for part in existing:
        print(f"  - {part}")
    
    if missing:
        print(f"\nMissing parts: {len(missing)}")
        for part in missing:
            print(f"  - {part}")
        print("\nDownload missing parts:")
        print("  cd index && python get_csv.py")
        return False
    
    print("\nAll parts downloaded!")
    return True


def extract_csv():
    print("\n" + "="*80)
    print("STEP 2: EXTRACTING CSV")
    print("="*80)
    
    csv_path = Path("index/data/news.csv")
    
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024**2)
        print(f"\nCSV already extracted: {size_mb:.1f} MB")
        return True
    
    print("\nExtracting from split zip archive...")
    print("This may take a few minutes...")
    
    # Try using zip command on macOS
    try:
        result = subprocess.run(
            ["zip", "-s", "0", "index/data/news.csv.zip", "--out", "index/data/news_full.zip"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            # Now extract the full zip
            result = subprocess.run(
                ["unzip", "-o", "index/data/news_full.zip", "-d", "index/data/"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("Extraction successful!")
                
                # Clean up
                Path("index/data/news_full.zip").unlink(missing_ok=True)
                
                if csv_path.exists():
                    size_mb = csv_path.stat().st_size / (1024**2)
                    print(f"CSV extracted: {size_mb:.1f} MB")
                    return True
        
        print(f"Error: {result.stderr}")
        return False
        
    except Exception as e:
        print(f"Error extracting: {e}")
        print("\nManual extraction needed:")
        print("  cd index/data")
        print("  zip -s 0 news.csv.zip --out news_full.zip")
        print("  unzip news_full.zip")
        return False


def sample_and_prepare_data(sample_size=1000):
    print("\n" + "="*80)
    print(f"STEP 3: PREPARING DATA (sampling {sample_size} articles)")
    print("="*80)
    
    import pandas as pd
    
    csv_path = Path("index/data/news.csv")
    
    if not csv_path.exists():
        print("ERROR: CSV not found")
        return None
    
    print("\nLoading CSV (this may take a moment)...")
    
    # Read only needed columns, sample to speed up
    try:
        # First, check file structure
        df_sample = pd.read_csv(csv_path, nrows=5)
        print(f"Columns available: {list(df_sample.columns)}")
        
        # Load full dataset with relevant columns
        # Adjust column names based on actual CSV structure
        usecols = None  # Will use all columns first
        
        print(f"Reading CSV...")
        df = pd.read_csv(csv_path, nrows=sample_size*10)  # Read more to filter
        
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Filter and prepare
        # Assuming columns: id, title, text/content, type/label
        # Adjust based on actual structure
        
        # Sample balanced dataset
        if 'type' in df.columns:
            label_col = 'type'
        elif 'label' in df.columns:
            label_col = 'label'
        else:
            print("Warning: No label column found, using all data")
            label_col = None
        
        if label_col:
            # Get balanced sample
            fake = df[df[label_col].str.lower().str.contains('fake', na=False)]
            reliable = df[df[label_col].str.lower().str.contains('reliable|credible|real', na=False)]
            
            n_per_class = min(sample_size // 2, len(fake), len(reliable))
            
            fake_sample = fake.sample(n=n_per_class, random_state=42)
            reliable_sample = reliable.sample(n=n_per_class, random_state=42)
            
            df_sampled = pd.concat([fake_sample, reliable_sample]).sample(frac=1, random_state=42)
        else:
            df_sampled = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # Save prepared data
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        
        train_path = output_dir / "train_sample.csv"
        test_path = output_dir / "test_sample.csv"
        
        # Split 80/20
        train_size = int(len(df_sampled) * 0.8)
        df_train = df_sampled.iloc[:train_size]
        df_test = df_sampled.iloc[train_size:]
        
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        
        print(f"\nData prepared:")
        print(f"  Train: {len(df_train)} articles -> {train_path}")
        print(f"  Test:  {len(df_test)} articles -> {test_path}")
        
        if label_col:
            print(f"\nTrain distribution:")
            print(df_train[label_col].value_counts())
            print(f"\nTest distribution:")
            print(df_test[label_col].value_counts())
        
        return str(train_path), str(test_path)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_index(train_csv):
    print("\n" + "="*80)
    print("STEP 4: BUILDING INDEX")
    print("="*80)
    
    print(f"\nBuilding index from: {train_csv}")
    print("This will take several minutes...")
    
    try:
        result = subprocess.run([
            sys.executable,
            "index/build_index_v3.py",
            "--input", train_csv,
            "--out-dir", "evaluation_store",
            "--batch-size", "128",
            "--index-type", "FlatIP"
        ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"Error building index:")
            print(result.stderr)
            return False
        
        print("\nIndex built successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def run_evaluation(test_csv, store_path):
    """Run full RAG vs Llama evaluation."""
    print("\n" + "="*80)
    print("STEP 5: RUNNING EVALUATION")
    print("="*80)
    
    print(f"\nTest data: {test_csv}")
    print(f"Store: {store_path}")
    
    try:
        result = subprocess.run([
            sys.executable,
            "evaluate/full_data_evaluation.py",
            "--data", test_csv,
            "--store", store_path,
            "--output", "evaluate/final_results"
        ], capture_output=False, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("\nEvaluation completed successfully!")
            print("\nResults saved in: evaluate/final_results/")
            return True
        else:
            print("Evaluation failed")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    print("COMPLETE RAG EVALUATION PIPELINE")
    print("Download -> Extract -> Prepare -> Build Index -> Evaluate")
    print()
    
    # Step 1: Check data
    if not check_data_downloaded():
        print("\nPlease download missing parts first:")
        print("  cd index && python get_csv.py")
        return
    
    # Step 2: Extract
    if not extract_csv():
        print("\nExtraction failed. Please extract manually.")
        return
    
    # Step 3: Prepare data
    result = sample_and_prepare_data(sample_size=1000)
    if not result:
        print("\nData preparation failed.")
        return
    
    train_csv, test_csv = result
    
    # Step 4: Build index
    if not build_index(train_csv):
        print("\nIndex building failed.")
        return
    
    # Step 5: Evaluate
    run_evaluation(test_csv, "evaluation_store")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nCheck results in: evaluate/final_results/")


if __name__ == "__main__":
    main()
