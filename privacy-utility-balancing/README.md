# Privacy-Utility Balancing

This package offers tools for speech deidentification, Equal Error Rate (EER) metric calculation, and dementia classification.

## Speech Deidentification

The deidentification process anonymizes speech files using five techniques:
1. **Pitch Shifting**
2. **Time Scaling**
3. **Adding Noise**
4. **Changing F0**
5. **Timbre Modification**

The input file, `Input.csv`, contains a column named `speech_path` that lists the file paths for the original WAV files. Each speech file is anonymized and stored in the `anonymized` folder, with "_anonymized" appended to the original file name. The output file, `Output.csv`, records the paths of the anonymized speech files in a column named `anonymized`.

## EER Calculation

The Equal Error Rate (EER) for speaker verification is calculated using the `librosa` package with Mel-Frequency Cepstral Coefficients (MFCC) features. The input file, `Input.csv`, requires three columns:
- `id`: Unique identifier for each speaker.
- `speech_path`: Path to the original speech file.
- `speech_anonymized_path`: Path to the corresponding anonymized speech file.

The EER is computed by comparing the MFCC features extracted from both the original and anonymized speech files.

## Dementia Classification

This package employs six classifiers to assess the utility of anonymized speech for dementia diagnosis:
- **Random Forest**
- **Support Vector Machine (SVM)**
- **k-Nearest Neighbors (kNN)**
- **Multi-Layer Perceptron (MLP)**
- **AdaBoost**
- **Gaussian Naive Bayes**

The classifiers are trained using the top 20 most important voice features to calculate the average classification accuracy, serving as the utility score. This score is then compared to that obtained from the original speech files.

### Input Files
- `Input.csv`: Contains the acoustic features of the speech files (rows) and a column named `DX` for the dementia diagnosis labels (NC, MCI, DE).
- `Input_features_importance.csv`: Lists all acoustic features in the `Feature` column, sorted by importance. The `num_important_features` column specifies the number of top-ranked features to be used in the classification task.

### Output
The classification accuracy for all six classifiers is computed using 10-fold cross-validation and printed for evaluation.
