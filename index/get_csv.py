import os
import shutil
import zipfile
from urllib import request as ur
from urllib.error import HTTPError, URLError
from zipfile import ZipFile
from dotenv import load_dotenv

# Load environment variables from params.env
load_dotenv('params.env')

DOWNLOAD_PAGE = "https://github.com/several27/FakeNewsCorpus/releases/download/v1.0"

# Use STORAGE_DIR from environment, or fallback to local data directory
STORAGE_DIR = os.getenv('STORAGE_DIR', '/StudentData/reproduce')
DATA_DIR = os.path.join(STORAGE_DIR, "data")

FILE_NAME = os.path.join(DATA_DIR, "news.csv")
PART_NAMES = ["news.csv.zip"] + [f"news.csv.z{i:02d}" for i in range(1, 10)]


def download(url, destination):
    if os.path.exists(destination):
        print(f"✓ Already exists: {os.path.basename(destination)}")
        return
    try:
        print(f"Downloading {os.path.basename(destination)}...", end="", flush=True)
        request = ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with ur.urlopen(request) as req, open(destination, "wb") as file:
            # Get file size if available
            file_size = req.headers.get('Content-Length')
            if file_size:
                file_size = int(file_size)
                downloaded = 0
                chunk_size = 8192
                while True:
                    chunk = req.read(chunk_size)
                    if not chunk:
                        break
                    file.write(chunk)
                    downloaded += len(chunk)
                    # Print progress
                    progress = (downloaded / file_size) * 100
                    print(f"\rDownloading {os.path.basename(destination)}: {progress:.1f}% ({downloaded / (1024**2):.1f}/{file_size / (1024**2):.1f} MB)", end="", flush=True)
                print()  # New line after completion
            else:
                # No size info, just copy
                shutil.copyfileobj(req, file)
                print(" Done")
        print(f"✓ Downloaded: {os.path.basename(destination)}")
    except (HTTPError, URLError) as e:
        print(f" Failed!")
        raise RuntimeError(f"Failed to download {url}: {e}")


def extract_split_zip(main_file, dest_dir):
    """Extract a multi-part zip archive."""
    import subprocess
    
    # Get directory and base name
    base_dir = os.path.dirname(main_file)
    base_name = os.path.basename(main_file)
    
    # Check if this is a split archive
    if base_name.endswith('.zip'):
        split_files = []
        # Check for .z01, .z02, etc.
        for i in range(1, 100):
            split_file = os.path.join(base_dir, f"news.csv.z{i:02d}")
            if os.path.exists(split_file):
                split_files.append(split_file)
            else:
                break
        
        if len(split_files) > 0:
            # This is a multi-part archive
            print(f"Found multi-part archive with {len(split_files) + 1} parts")
            print("Joining parts in correct order...")
            
            # Concatenate all parts in order: z01, z02, ..., z09, then .zip
            combined_zip = os.path.join(base_dir, "full_news.csv.zip")
            
            with open(combined_zip, 'wb') as outfile:
                # Add all .z## files in order
                for split_file in sorted(split_files):
                    print(f"  Adding {os.path.basename(split_file)}...")
                    with open(split_file, 'rb') as infile:
                        outfile.write(infile.read())
                # Finally add the .zip file
                print(f"  Adding {base_name}...")
                with open(main_file, 'rb') as infile:
                    outfile.write(infile.read())
            
            print("Extracting combined archive...")
            # Use unzip command instead of Python's zipfile
            result = subprocess.run(
                ['unzip', '-o', combined_zip],
                cwd=base_dir,
                capture_output=True,
                text=True
            )
            
            # Clean up combined file
            os.remove(combined_zip)
            
            # Check if file was extracted (ignore error codes, unzip often warns but succeeds)
            csv_file = os.path.join(dest_dir, "news.csv")
            if not csv_file.startswith(dest_dir):
                # Try finding it in base_dir instead
                csv_file = os.path.join(base_dir, "news_cleaned_2018_02_13.csv")
                if os.path.exists(csv_file):
                    # Rename to expected name
                    import shutil
                    shutil.move(csv_file, os.path.join(dest_dir, "news.csv"))
                    print("✓ Extraction successful! (file renamed)")
                    return
            
            if os.path.exists(os.path.join(dest_dir, "news.csv")):
                print("✓ Extraction successful!")
                return
            
            # Check for the actual extracted filename
            extracted_file = os.path.join(base_dir, "news_cleaned_2018_02_13.csv")
            if os.path.exists(extracted_file):
                import shutil
                shutil.move(extracted_file, os.path.join(dest_dir, "news.csv"))
                print("✓ Extraction successful! (file renamed)")
                return
            
            # If we get here, extraction truly failed
            raise RuntimeError(f"Extraction failed - no CSV file found")
    
    # Regular single zip file
    with ZipFile(main_file, "r") as zip:
        zip.extractall(dest_dir)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(FILE_NAME) and os.path.getsize(FILE_NAME) > 0:
        file_size = os.path.getsize(FILE_NAME) / (1024**3)  # GB
        print(f"✓ CSV file already present at {FILE_NAME} ({file_size:.2f} GB)")
        return

    # Check if all parts are already downloaded
    all_parts_exist = all(os.path.exists(os.path.join(DATA_DIR, part)) for part in PART_NAMES)
    
    if not all_parts_exist:
        print(f"Downloading FakeNews Corpus to: {DATA_DIR}")
        print(f"This will download {len(PART_NAMES)} files and extract the CSV...")
        print()

        local_paths = []
        for i, part_name in enumerate(PART_NAMES, 1):
            print(f"[{i}/{len(PART_NAMES)}] ", end="")
            url = f"{DOWNLOAD_PAGE}/{part_name}"
            destination = os.path.join(DATA_DIR, part_name)
            download(url, destination)
            local_paths.append(destination)
    else:
        print(f"✓ All {len(PART_NAMES)} archive parts already downloaded")
        local_paths = [os.path.join(DATA_DIR, part) for part in PART_NAMES]

    # Extract the zip file
    print()
    print("Extracting news.csv from split zip archive...")
    try:
        extract_split_zip(local_paths[0], DATA_DIR)
        print(f"✓ Extraction complete!")
        if os.path.exists(FILE_NAME):
            file_size = os.path.getsize(FILE_NAME) / (1024**3)  # GB
            print(f"✓ CSV file ready at: {FILE_NAME} ({file_size:.2f} GB)")
        else:
            print(f"❌ Error: CSV file not found after extraction at {FILE_NAME}")
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        print("Please check if all archive parts downloaded correctly.")


if __name__ == "__main__":
    main()
