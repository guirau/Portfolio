# Dataset Preprocessing Pipeline

Downloads and organizes recipe images and their masks from Google Drive, following MediaValet naming conventions.

## Pipeline Steps

1. Downloads images from Google Drive
2. Extracts metadata and keywords
3. Validates filename structure
4. Organizes into structured dataset
5. Maintains image-mask pairs integrity

## File Structure
```
dataset_prep/
├── metadata.csv            # Image metadata
├── invalid_filenames.txt   # Invalid files log
├── tmp/                    # Raw downloads
│   ├── original/
│   └── mask/
└── ImageFolder/            # Organized dataset
    ├── original/
    └── mask/
```

## Filename Format

`BRAND_Y##_IMAGETYPE_W##_COUNTRY_RECIPECODE_##_PARAM_[original|mask].jpg`

Example: `KN_Y22_STEP_W12_UK_RC123_01_PREP_original.jpg`

## Metadata Format

The pipeline generates a `metadata.csv` file with parsed filename components and image keywords:

| Column | Description | Example |
|--------|-------------|---------|
| FileName | Complete filename | HF_Y24_R01_W48_UK_FR1646-37_05__original.jpg |
| Brand | Company identifier | HF |
| Year | Year code | Y24 |
| ImageType | Type of image | R01 |
| Week | Week number | W48 |
| Country | Country code | UK |
| RecipeCode | Recipe identifier | FR1646-37 |
| Step | Step number or Main | 05 |
| ExtraParam | Additional parameters | |
| Keywords | Image keywords | ['pan' 'steplibrary'] |


## Key Functions

- `main()`: Orchestrates download and metadata extraction
- `sort_images_by_keyword()`: Organizes images by categories
- `check_and_download_missing_pairs()`: Ensures image-mask pair completeness

## Requirements

- exiftool: Metadata extraction
- pandas, numpy: Data handling
- Google Drive API: File access
- loguru: Logging

## Usage

```python
# Download images
main(gdrive_folder_id, output_csv, download_dir, invalid_filenames_txt)

# Sort into categories
sort_images_by_keyword(input_folder, output_folder, csv_metadata, 
                      relevant_keywords, keyword_mapping)
```
