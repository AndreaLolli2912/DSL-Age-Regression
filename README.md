# DSL-Age-Estimation

A Machine Learning project for the **_Data Science Lab: process and methods_** class, **Age Estimation** using both **tabular** and **raw audio** features. Developed in collaboration with  ![Daniele Famà](@daniele), this repository demonstrates the end-to-end process of data loading, feature extraction, preprocessing, and model training with advanced AI regression techniques.

---

## Highlights

- **AI Techniques**: Utilizes **CatBoost**, **Random Forest**, and other ML algorithms for robust regression.  
- **Audio Feature Extraction**: Gathers spectral, Mel-spectrogram, and time-domain statistics (skew, kurtosis, etc.) to capture essential speech cues.  
- **Tabular Feature Engineering**: Cleans and preprocesses demographic and acoustic-linguistic data (outlier detection, scaling, encoding).  
- **Model Selection & Tuning**: Employs dimensionality reduction (e.g., LDA) and hyperparameter search for optimal performance.  
- **Result**: Achieved competitive RMSE scores on the public leaderboard, showing effectiveness of the combined approach.

For a detailed description of the methods, refer to the accompanying [report](./Fama_Lolli_Age_Regression.pdf).

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YourGitHubUsername/DSL-Age-Estimation.git
   cd DSL-Age-Estimation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Folder Structure

```
DSL-Age-Estimation
├── catboost_info
├── data
│   ├── audios_development
│   ├── audio_evaluation
│   ├── development.csv
│   └── evaluation.csv
├── figs
├── submissions
├── utils
│   └── audio_extraction.py
├── audio_features_preprocessing.ipynb
├── main.ipynb
├── report_figures.ipynb
├── Fama_Lolli_Age_Regression.pdf
├── README.md
└── requirements.txt
```

- **`data`**: Place all provided `.csv` files and audio folders here.  
- **`utils/audio_extraction.py`**: Extracts raw audio features (adjust lowcut, highcut, top_db as needed).  
- **`audio_features_preprocessing.ipynb`**: Converts raw audio features into tabular format (statistical summaries).  
- **`main.ipynb`**: Runs the entire pipeline (EDA, model training, evaluation).

---

## Usage

1. **Extract Raw Audio Features**  
   ```bash
   python main/utils/audio_extraction.py
   ```
   Adjust cleaning parameters (e.g., `lowcut`, `highcut`, `top_db`) as desired.

2. **Preprocess Features**  
   Open and run **`audio_features_preprocessing.ipynb`** to produce summarized data in `.h5` or `.csv` formats.

3. **Run the Model Pipeline**  
   Use **`main.ipynb`** to orchestrate data loading, EDA, and model training with CatBoost/Random Forest.

---

## Credits

- Developed by [@YourGitHubUsername] in collaboration with [@CoStudentGitHubUsername].  
- For full technical details, please refer to our [report](./Fama_Lolli_Age_Regression.pdf).

Enjoy experimenting with **DSL-Age-Estimation**! Feel free to open issues or submit pull requests for improvements.
