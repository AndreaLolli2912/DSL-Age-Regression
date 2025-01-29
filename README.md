# DSL-Age-Estimation

## 1. Install Dependencies

Install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

## 2. Folder Organization

Organize your data within the main folder using the following hierarchy:

main/data
│
├── audios_development
├── audio_evaluation
├── development.csv
└── evaluation.csv

## 3. Extract Raw Audio Features and Save as .h5 Format

Run the Python script to extract raw audio features:

```bash
python main/utils/audio_extraction.py
```

You can adjust the cleaning settings by tweaking the following values in the script:
- lowcut
- highcut
- top_db

## 4. Preprocess Raw Audio Features into Tabular Format

Convert the extracted raw audio features into a tabular format with comprehensive statistical summaries by running the notebook:

main/audio_features_preprocessing.ipynb. 

It will save the new dataframes in the folder main/data

## 5. Run the Model Pipeline

Visualize and execute the entire model pipeline, from data loading to exploratory data analysis (EDA) and model selection, by running the main notebook:

main/main.ipynb