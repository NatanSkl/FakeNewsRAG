import os
import shutil
import zipfile
from urllib import request as ur
from urllib.error import HTTPError, URLError
from zipfile import ZipFile

DOWNLOAD_PAGE = "https://github.com/several27/FakeNewsCorpus/releases/download/v1.0"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FILE_NAME = os.path.join(DATA_DIR, "news.csv")
PART_NAMES = ["news.csv.zip"] + [f"news.csv.z{i:02d}" for i in range(1, 10)]


def download(url, destination):
    if os.path.exists(destination):
        return
    try:
        request = ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with ur.urlopen(request) as req, open(destination, "wb") as file:
            shutil.copyfileobj(req, file)
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def extract_split_zip(main_file, dest_dir):
    with ZipFile(main_file, "r") as zip:
        zip.extractall(dest_dir)
        # for info in zip.infolist():
        #     if info.is_dir():
        #         continue
        #     out_path = os.path.join(dest_dir, info.filename)
        #     os.makedirs(os.path.dirname(out_path), exist_ok=True)
        #     zip.extract(info, dest_dir)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(FILE_NAME) and os.path.getsize(FILE_NAME) > 0:
        print("CSV file already present at {}".format(FILE_NAME))
        return

    local_paths = []
    for part_name in PART_NAMES:
        url = f"{DOWNLOAD_PAGE}/{part_name}"
        destination = os.path.join(DATA_DIR, part_name)
        download(url, destination)
        local_paths.append(destination)

    # FIXME: Handle split zips. For now extract manually
    try:
        extract_split_zip(local_paths[0], DATA_DIR)
    except zipfile.BadZipfile:
        # TODO: remove, temporary local fix for split zips
        os.system(f"open -a 'The Unarchiver.app' {local_paths[0]}")


if __name__ == "__main__":
    main()
