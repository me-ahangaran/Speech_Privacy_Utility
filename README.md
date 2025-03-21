# Speech Deidentification with Preservation of Cognitive Acoustic Feature Utility

This repository contains the implementation of a speech deidentification approach that balances privacy and utility. It includes various packages for feature engineering, speaker deidentification, privacy metrics, and classification tasks for dementia diagnosis.

<image src = "https://github.com/me-ahangaran/Speech_Privacy_Utility/blob/main/Flowchart.svg">

## Feature Engineering

This package provides tools for extracting acoustic features and calculating the importance of cognitive features. Specifically:

- **Acoustic Features Extraction**: Extracts 481 statistical vocal measures derived from 12 sets of acoustic variables.
- **Feature Importance Calculation**: Utilizes a Random Forest Regression model with an ensemble of 100 decision tree estimators to assess the importance of various voice features.

## Privacy-Utility Balancing

This package includes tools for speech deidentification, privacy metric calculation, and dementia classification:

- **Speech Deidentification**: Employs methods such as pitch shifting at various levels, time scaling, noise addition, F0 modification, and timbre alteration. The deidentification process can be customized based on different privacy levels through parameter settings.
- **Privacy Metric Calculation**: Uses the Equal Error Rate (EER) metric to quantify the privacy level achieved by the deidentification process.
- **Dementia Classification**: Implements six advanced classification algorithms to assess the accuracy of cognitive impairment diagnosis using deidentified speech data.

## Experimental Results on the FHS and Dementia Bank Cohorts

This section describes our evaluation of the framework's performance in **de-identification** and **cognitive utility preservation** when applied to voice data from the **Framingham Heart Study (FHS)** and the **Dementia Bank Delaware corpus**.

### Dataset Information

- **FHS Dataset**: Comprises 128 speech samples from neuropsychological examinations of participants with varying levels of cognitive impairment:
  - **NC** (Normal Cognition)
  - **MCI** (Mild Cognitive Impairment)
  - **DE** (Dementia)
  
- **Dementia Bank Dataset**: Includes 85 speech samples with participants categorized as:
  - **NC** (Normal Cognition)
  - **MCI** (Mild Cognitive Impairment)

The table below shows the results of applying the framework to these datasets, before and after de-identification, using **six classification algorithms** to assess accuracy.

<image src = "https://github.com/me-ahangaran/Speech_Privacy_Utility/blob/main/Orig_Altered_Results_Table.jpg" >

### Processing Steps

Each speech file includes transcriptions with the exact timing and duration of clinician-participant interactions. Using these transcriptions, we applied **speech diarization** to isolate the participant's speech segments, then merged these segments into a single file containing only the participant's voice. This diarized version (excluding the clinician’s voice) was used for all further processing, ensuring that analysis for cognitive impairment differentiation and speech de-identification focuses solely on participants' vocal features.

### Results and Metrics

We evaluated the model performance with increasing **pitch-shifting** levels, assessing metrics such as:
- **Equal Error Rate (EER)**
- **Average Classification Accuracy**
- **AUC** of the speaker recognition system

These results are provided for both datasets. The optimal balancing points for both datasets are marked by vertical orange dashed lines.‎

<image src = "https://github.com/me-ahangaran/Speech_Privacy_Utility/blob/main/FHS_DementiaBank_Results.jpg" >


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
