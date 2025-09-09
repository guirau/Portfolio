"""Downloads images from a Google Drive folder and saves metadata to a CSV file."""

import itertools
import os
import re
import shutil
import time

import exiftool as exiftool
import numpy as np
import pandas as pd
from loguru import logger
from oma_recipeclassifier.utils.google_drive import GoogleDrive

# Regex pattern to validate filenames
FILENAME_PATTERN = re.compile(
    r"^(?P<brand>[A-Z]{2})_"  # Brand: 2 capital letters
    r"(?P<year>Y\d{2})_"  # Year: 'Y' followed by 2 digits
    r"(?P<image_type>([A-Z\d])+(_[A-AC-VX-Z\d]+)?)_"  # ImageType: 1+ capital letters or digis, potentially separated by '_'
    r"(?P<week>[BW]+\d{2})_"  # Week: 'W' or 'WB' followed by 2 digits
    r"(?P<country>[A-Z]{2,4})_"  # Country: 2-4 uppercase letters
    r"(?P<recipe_code>[A-Z\d-]+)?"  # Recipe Code: 1+ alphanumeric characters or '-' (optional)
    r"(?P<spacer_1>_)?"  # Spacer: Optional '_'
    r"(?P<extra_param_edge_case>[A-Z]+)?"  # ExtraParameter (edge case): 1+ uppercase letters (optional)
    r"(?P<spacer_2>_)?"  # Spacer: Optional '_'
    r"(?P<step>\d{2}|main|Main|MAIN)_"  # Step: 2 digits (01, 02...) or 'main' in any case format
    r"(?P<extra_param>[A-Z\d ]+)?"  # ExtraParameter: 1+ uppercase letters digits, or spaces (optional)
    r"(?P<spacer_3>[_]+)?"  # Spacer: 1+ optional '_'
    r"(?P<tag>original|mask).jpg$"  # Tag: original or mask, with .jpg format
)


def extract_keywords(image_path):
    """Extracts keywords from image metadata."""
    try:
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(image_path)
            keywords = metadata[0].get("IPTC:Keywords")
            if isinstance(keywords, list):
                return keywords
            else:
                return [keywords]
    except Exception as ex:
        logger.error(f"Error extracting metadata from image: {image_path}: {ex}")
    return []


def extract_from_filename(filename):
    """Extracts information from a filename following MediaValet regex pattern."""
    match = FILENAME_PATTERN.match(filename)
    if not match:
        logger.warning(
            f"Filename {filename} doesn't follow MediaValet naming convention."
        )
        return (None,) * 9

    brand = match.group("brand")
    year = match.group("year")
    image_type = match.group("image_type")
    week = match.group("week")
    country = match.group("country")
    recipe_code = match.group("recipe_code")
    step = match.group("step")
    extra_param = match.group("extra_param")

    return (
        brand,
        year,
        image_type,
        week,
        country,
        recipe_code,
        step,
        extra_param,
    )


def list_missing_images(folder_id, existing_filenames, output_txt):
    """Yields missing images from Google Drive that are not present in the CSV file."""
    files = drive.list_files_in_folder(folder_id)
    logger.info(f"Total number of JPEG files found in folder: {len(files)}")

    invalid_filenames = []  # List to store filenames not matching naming convention
    already_present_files = []  # List to store filenames already present in CSV
    matching_count = 0

    for file in files:
        if file["name"].lower().endswith("original.jpg") or file[
            "name"
        ].lower().endswith("mask.jpg"):
            if file["name"].replace("mask", "original") not in existing_filenames:
                if FILENAME_PATTERN.match(file["name"]):
                    matching_count += 1
                    yield file
                else:
                    invalid_filenames.append(file["name"])
            else:
                already_present_files.append(file["name"])

    # Write invalid filenames to a txt file
    with open(output_txt, "w") as f:
        for invalid_filename in invalid_filenames:
            f.write(f"{invalid_filename}\n")

    logger.info(
        f"Total number of missing files matching MediaValet naming convention: {matching_count}"
    )
    logger.info(
        f"Total number of missing files not matching MediaValet naming convention: {len(invalid_filenames)}"
    )


def download_image(file, download_dir):
    """Downloads a single image from Google Drive."""
    subfolder = "main" if "main" in file["name"].lower() else "step"
    if file["name"].lower().endswith("original.jpg"):
        download_path = os.path.join(download_dir, "original", subfolder)
    elif file["name"].lower().endswith("mask.jpg"):
        download_path = os.path.join(download_dir, "mask", subfolder)
    else:
        return None

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    file_path = drive.download_file(file["id"], file["name"], download_path)
    logger.info(f"Downloaded {file['name']} to: {file_path}")
    return file_path


def write_row_to_csv(row_data, csv_path):
    """Writes a single row to the CSV file."""
    df = pd.DataFrame([row_data])

    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, mode="w", index=False, header=True)
    else:
        df.to_csv(csv_path, mode="a", index=False, header=False)

    logger.success(f"Added row for {row_data['FileName']} to CSV.")


def delete_local_file(file_path):
    """Deletes a local file from the specified path."""
    try:
        os.remove(file_path)
        logger.info(f"Deleted {file_path} from local directory.")
    except Exception as ex:
        logger.error(f"Error deleting {file_path}: {ex}")


def check_and_download_missing_pairs(folder_id, download_dir):
    """Cheks if any original/mask pairs are missing and downloads them."""
    drive_files = drive.list_files_in_folder(folder_id)
    originals = set()
    masks = set()

    # Collect original and mask files from download directory
    for root, _, files in os.walk(download_dir):
        for filename in files:
            if filename.endswith("original.jpg"):
                originals.add(filename)
            elif filename.endswith("mask.jpg"):
                masks.add(filename)

    # Check missing pairs
    missing_masks = [
        f.replace("original.jpg", "mask.jpg")
        for f in originals
        if f.replace("original.jpg", "mask.jpg") not in masks
    ]
    missing_originals = [
        f.replace("mask.jpg", "original.jpg")
        for f in masks
        if f.replace("mask.jpg", "original.jpg") not in originals
    ]

    if not missing_masks and not missing_originals:
        logger.info("No files are missing. All original/mask pairs are present.")
    else:
        # Download missing pairs if available in Google Drive files
        for missing_mask in missing_masks:
            logger.info(
                f"Missing mask for {missing_mask.replace('_mask.jpg', '_original.jpg')}. Downloading..."
            )
            file = next((f for f in drive_files if f["name"] == missing_mask), None)
            if file:
                download_image(file, download_dir)

        for missing_original in missing_originals:
            logger.info(
                f"Missing original for {missing_original.replace('_original.jpg', '_mask.jpg')}. Downloading..."
            )
            file = next((f for f in drive_files if f["name"] == missing_original), None)
            if file:
                download_image(file, download_dir)


def main(folder_id, output_csv, download_dir, invalid_filenames_output_txt):
    """Main function to download missing images and update the CSV file."""
    # Load CSV and get existing filenames
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        existing_filenames = set(df["FileName"])
    else:
        existing_filenames = set()

    # List missing images from Google Drive
    missing_images = list_missing_images(
        folder_id, existing_filenames, invalid_filenames_output_txt
    )
    missing_images, missing_images_copy = itertools.tee(
        missing_images
    )  # _copy just for logging
    total_missing_images = len(list(missing_images_copy))
    logger.info(f"Total number of missing files: {total_missing_images}")

    input("Press Enter to start downloading missing images...")

    total_download_time, downloaded_images = 0, 0  # To log time

    # Check for missing pairs and download them if present in Google Drive
    logger.info("Checking if any original/mask file pairs are missing.")
    check_and_download_missing_pairs(folder_id, download_dir)

    # Download each missing image and update CSV
    for file in missing_images:
        start_time = time.time()  # To log time
        file_path = download_image(file, download_dir)
        download_time = time.time() - start_time  # To log time

        # Extract metadata and update CSV only for 'original' images
        if file_path:
            total_download_time += download_time
            downloaded_images += 1

            if file["name"].lower().endswith("original.jpg"):
                keywords = extract_keywords(file_path)
                (
                    brand,
                    year,
                    image_type,
                    week,
                    country,
                    recipe_code,
                    step,
                    extra_param,
                ) = extract_from_filename(file["name"])
                # Create a dict with the extracted data
                image_data = {
                    "FileName": file["name"],
                    "Brand": brand,
                    "Year": year,
                    "ImageType": image_type,
                    "Week": week,
                    "Country": country,
                    "RecipeCode": recipe_code,
                    "Step": step,
                    "ExtraParam": extra_param,
                    "Keywords": np.array(list(set(keywords))) if keywords else None,
                }

                write_row_to_csv(image_data, output_csv)

            # To log time
            if downloaded_images > 0:
                avg_download_time = total_download_time / downloaded_images
                remaining_images = total_missing_images - downloaded_images
                estimated_time_left = avg_download_time * remaining_images
                minutes, seconds = divmod(estimated_time_left, 60)
                logger.info(f"Estimated time left: {int(minutes)}min {int(seconds)}s")


def sort_images_by_keyword(
    input_folder, output_folder, csv_metadata, relevant_keywords, keyword_mapping
):
    """Sorts images and masks into separate categories based on their keywords."""
    # Load metadata CSV
    if not os.path.exists(csv_metadata):
        logger.error(f"Metadata CSV file not found at: {csv_metadata}")
        return

    metadata = pd.read_csv(csv_metadata)

    def _normalize_keywords(keywords):
        if isinstance(keywords, np.ndarray):
            return keywords.tolist()
        elif isinstance(keywords, str):
            return keywords.strip("[]").replace("'", "").replace('"', "").split()
        return keywords

    metadata["Keywords"] = metadata["Keywords"].apply(_normalize_keywords)

    # Log. Total number of files in input folder
    total_files = sum([len(files) for _, _, files in os.walk(input_folder)])
    logger.info(f"Total number of files in input folder: {total_files}")
    input("Press Enter to start sorting out the dataset...")

    # Create output folder structure
    original_dir = os.path.join(output_folder, "original")
    mask_dir = os.path.join(output_folder, "mask")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Log. Counters for logging
    total_relevant_files_metadata = 0
    total_relevant_files = 0
    total_copied_originals = 0
    total_copied_masks = 0

    # Iterate over metadata and organize files
    for _, row in metadata.iterrows():
        file_name = row["FileName"]
        keywords = row["Keywords"]

        # Log
        if isinstance(keywords, list) and any(
            keyword in relevant_keywords for keyword in keywords
        ):
            source_paths = [
                os.path.join(input_folder, subfolder, subsubfolder, file_name)
                for subfolder in ["original", "mask"]
                for subsubfolder in ["main", "step"]
            ]
            if any(os.path.exists(path) for path in source_paths):
                total_relevant_files += 1

        # Find the class for the image
        category = None
        for keyword in keywords:
            if keyword in relevant_keywords:
                category = keyword_mapping.get(keyword, keyword)
                break

        if not category:
            logger.warning(f"No relevant category found for {file_name}, skipping.")
            continue

        total_relevant_files_metadata += 1  # Log

        # Determine source and destination paths
        for subfolder in ["original", "mask"]:
            if subfolder == "mask":
                file_name = file_name.replace("_original", "_mask")

            category_path = os.path.join(output_folder, subfolder, category)
            os.makedirs(category_path, exist_ok=True)

            source_path_main = os.path.join(input_folder, subfolder, "main", file_name)
            source_path_step = os.path.join(input_folder, subfolder, "step", file_name)
            destination_path = os.path.join(category_path, file_name)

            if os.path.exists(source_path_main):
                shutil.copy(source_path_main, destination_path)
                logger.success(f"Copied {file_name} to {destination_path}.")
                if subfolder == "original":  # Log
                    total_copied_originals += 1
                elif subfolder == "mask":  # Log
                    total_copied_masks += 1
            elif os.path.exists(source_path_step):
                shutil.copy(source_path_step, destination_path)
                logger.success(f"Copied {file_name} to {destination_path}.")
                if subfolder == "original":  # Log
                    total_copied_originals += 1
                elif subfolder == "mask":  # Log
                    total_copied_masks += 1
            else:
                logger.warning(
                    f"Source file {file_name} not found in either 'main' or 'step' subfolders. Skipping."
                )

    # Logging the final counts
    logger.success(
        f"Total number of files in metadata.csv with relevant keywords: {total_relevant_files_metadata}"
    )
    logger.success(
        f"Total number of files in input folder (images+masks): {total_files}"
    )
    logger.success(
        f"Total number of files in input folder with relevant keywords (images+masks): {total_relevant_files * 2}"
    )
    logger.success(f"Total number of original files copied: {total_copied_originals}")
    logger.success(f"Total number of mask files copied: {total_copied_masks}")


if __name__ == "__main__":
    gdrive_folder_id = "1XJTUZPH89dUcdmblgTjgAksqWeThGdQG"
    output_csv = os.path.join("oma_recipeclassifier", "v0", "dataset", "metadata.csv")
    download_dir = os.path.join("oma_recipeclassifier", "v0", "dataset", "tmp")
    invalid_filenames_output_txt = os.path.join(
        "oma_recipeclassifier", "v0", "dataset", "invalid_filenames.txt"
    )

    drive = GoogleDrive()

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    ##### DOWNLOAD ######
    main(gdrive_folder_id, output_csv, download_dir, invalid_filenames_output_txt)

    # Create ImageFolder like structure for dataset
    input_folder = os.path.join("oma_recipeclassifier", "v0", "dataset", "tmp")
    output_folder = os.path.join("oma_recipeclassifier", "v0", "dataset", "ImageFolder")
    csv_metadata = output_csv

    relevant_keywords = [
        "CP",
        "glass-bowl-large",
        "glass-bowl-small",
        "chopping-board",
        "oven-dish",
        "sauce-pan",
        "pot-one-handle",
        "pot-two-handles-medium",
        "saucepan",
        "glass-bowl-medium",
        "pot-two-handles-small",
        "oven-tray",
        "group_step",
        "pan",
        "pot-two-handles-shallow",
    ]

    keyword_mapping = {
        # "mapped keyword": "destination category"
        "sauce-pan": "saucepan",
    }

    # ##### SORT OUT ######
    # sort_images_by_keyword(
    #     input_folder, output_folder, csv_metadata, relevant_keywords, keyword_mapping
    # )
