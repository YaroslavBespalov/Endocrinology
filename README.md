# Endocrinology

## Dataset Description

In general, we have a dataset composed of ultrasound (US) `Images` of the thyroid and the corresponding labeled formations (for each patient there may be several different types) - `Masks`. You need to place patients data with the following hierarchy:

For each patient you need to place data with the following hierarchy:

```
.
├── path_to_raw_data
│   ├── Patient_name
│       ├── Images
│           ├── image_name.tif
│           ├── image_name.tif
│           ...
│       └── Masks
│           ├── mask_name.tif
│           ├── mask_name.tif
│           ...
│   ├── Patient_name
│       ├── Images
│       └── Masks
│   ...
```

## Data Preprocessing

To remove all unnecessary information from the frames and prepare the data for model you should run the python script `data_preprocessing.py`:

Example of usage:

```
python ./data_preprocessing.py --path_to_raw_data=<str> --path_to_preprocessed_data=<str>
```

As a result, you will receive a `path_to_preprocessed_data` folder with prepared images with the same structure as the `path_to_raw_data` folder, as well as a csv table `dataset.csv` with a description of the dataset for convenient use in the **data loader** further.
