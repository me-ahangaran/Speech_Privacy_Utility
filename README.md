# Speech Deidentification with Preservation of Cognitive Acoustic Feature Utility

This repository contains the implementation of a speech deidentification approach that balances privacy and utility. It includes various packages for feature engineering, speaker deidentification, privacy metrics, and classification tasks for dementia diagnosis.

## Feature Engineering

This package provides tools for extracting acoustic features and calculating the importance of cognitive features. Specifically:

- **Acoustic Features Extraction**: Extracts 481 statistical vocal measures derived from 12 sets of acoustic variables.
- **Feature Importance Calculation**: Utilizes a Random Forest Regression model with an ensemble of 100 decision tree estimators to assess the importance of various voice features.

## Privacy-Utility Balancing

This package includes tools for speech deidentification, privacy metric calculation, and dementia classification:

- **Speech Deidentification**: Employs methods such as pitch shifting at various levels, time scaling, noise addition, F0 modification, and timbre alteration. The deidentification process can be customized based on different privacy levels through parameter settings.
- **Privacy Metric Calculation**: Uses the Equal Error Rate (EER) metric to quantify the privacy level achieved by the deidentification process.
- **Dementia Classification**: Implements six advanced classification algorithms to assess the accuracy of cognitive impairment diagnosis using deidentified speech data.

## Package Requirements

The following packages and versions are required for running the toolboxes:

```bash
  - librosa 0.10.1
  - numpy 1.26.4
  - pandas 1.5.3
  - sklearn 1.2.2
  - soundfile 0.12.1
  - os
  - pyworld 0.3.4
