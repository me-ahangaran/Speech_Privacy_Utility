""" We employed Random Forest Regression with an ensemble of 100 decision tree estimators to calculate the importance of voice features. 
The importance values were then normalized to the interval [0, 1]. 
Finally, the features and their corresponding importance scores were saved as a CSV file for output.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


voice_features = ['amp_min','amp_max', 'amp_range', 'amp_mean', 'amp_std', 'amp_median', 'amp_median_min', 'amp_median_max',
              'rms_min', 'rms_max', 'rms_range', 'rms_mean', 'rms_std', 'rms_median', 'rms_median_min', 'rms_median_max',
              'poly_features_min', 'poly_features_max', 'poly_features_range', 'poly_features_mean',
              'poly_features_std', 'poly_features_median', 'poly_features_median_min', 'poly_features_median_max',
              'spectral_bandwidth_min', 'spectral_bandwidth_max', 'spectral_bandwidth_range', 'spectral_bandwidth_mean', 
              'spectral_bandwidth_std', 'spectral_bandwidth_median', 'spectral_bandwidth_median_min', 'spectral_bandwidth_median_max',
              'spectral_centroid_min', 'spectral_centroid_max', 'spectral_centroid_range', 'spectral_centroid_mean', 
              'spectral_centroid_std', 'spectral_centroid_median', 'spectral_centroid_median_min', 'spectral_centroid_median_max',
              'spectral_flatness_min', 'spectral_flatness_max', 'spectral_flatness_range', 'spectral_flatness_mean', 
              'spectral_flatness_std', 'spectral_flatness_median', 'spectral_flatness_median_min', 'spectral_flatness_median_max',
              'spectral_rolloff_min', 'spectral_rolloff_max', 'spectral_rolloff_range', 'spectral_rolloff_mean', 
              'spectral_rolloff_std', 'spectral_rolloff_median', 'spectral_rolloff_median_min', 'spectral_rolloff_median_max',
              'zero_crossing_rate_min', 'zero_crossing_rate_max', 'zero_crossing_rate_range', 'zero_crossing_rate_mean', 
              'zero_crossing_rate_std', 'zero_crossing_rate_median', 'zero_crossing_rate_median_min', 'zero_crossing_rate_median_max',
              'tempo',
              'chroma_cens_0_min', 'chroma_cens_0_max', 'chroma_cens_0_range', 'chroma_cens_0_mean',
              'chroma_cens_0_std', 'chroma_cens_0_median', 'chroma_cens_0_median_min', 'chroma_cens_0_median_max',
              'chroma_cens_1_min', 'chroma_cens_1_max', 'chroma_cens_1_range', 'chroma_cens_1_mean',
              'chroma_cens_1_std', 'chroma_cens_1_median', 'chroma_cens_1_median_min', 'chroma_cens_1_median_max',
              'chroma_cens_2_min', 'chroma_cens_2_max', 'chroma_cens_2_range', 'chroma_cens_2_mean',
              'chroma_cens_2_std', 'chroma_cens_2_median', 'chroma_cens_2_median_min', 'chroma_cens_2_median_max',
              'chroma_cens_3_min', 'chroma_cens_3_max', 'chroma_cens_3_range', 'chroma_cens_3_mean',
              'chroma_cens_3_std', 'chroma_cens_3_median', 'chroma_cens_3_median_min', 'chroma_cens_3_median_max',
              'chroma_cens_4_min', 'chroma_cens_4_max', 'chroma_cens_4_range', 'chroma_cens_4_mean',
              'chroma_cens_4_std', 'chroma_cens_4_median', 'chroma_cens_4_median_min', 'chroma_cens_4_median_max',
              'chroma_cens_5_min', 'chroma_cens_5_max', 'chroma_cens_5_range', 'chroma_cens_5_mean',
              'chroma_cens_5_std', 'chroma_cens_5_median', 'chroma_cens_5_median_min', 'chroma_cens_5_median_max',
              'chroma_cens_6_min', 'chroma_cens_6_max', 'chroma_cens_6_range', 'chroma_cens_6_mean',
              'chroma_cens_6_std', 'chroma_cens_6_median', 'chroma_cens_6_median_min', 'chroma_cens_6_median_max',
              'chroma_cens_7_min', 'chroma_cens_7_max', 'chroma_cens_7_range', 'chroma_cens_7_mean',
              'chroma_cens_7_std', 'chroma_cens_7_median', 'chroma_cens_7_median_min', 'chroma_cens_7_median_max',
              'chroma_cens_8_min', 'chroma_cens_8_max', 'chroma_cens_8_range', 'chroma_cens_8_mean',
              'chroma_cens_8_std', 'chroma_cens_8_median', 'chroma_cens_8_median_min', 'chroma_cens_8_median_max',
              'chroma_cens_9_min', 'chroma_cens_9_max', 'chroma_cens_9_range', 'chroma_cens_9_mean',
              'chroma_cens_9_std', 'chroma_cens_9_median', 'chroma_cens_9_median_min', 'chroma_cens_9_median_max',
              'chroma_cens_10_min', 'chroma_cens_10_max', 'chroma_cens_10_range', 'chroma_cens_10_mean',
              'chroma_cens_10_std', 'chroma_cens_10_median', 'chroma_cens_10_median_min', 'chroma_cens_10_median_max',
              'chroma_cens_11_min', 'chroma_cens_11_max', 'chroma_cens_11_range', 'chroma_cens_11_mean',
              'chroma_cens_11_std', 'chroma_cens_11_median', 'chroma_cens_11_median_min', 'chroma_cens_11_median_max',
              'mfcc_0_min', 'mfcc_0_max', 'mfcc_0_range', 'mfcc_0_mean',
              'mfcc_0_std', 'mfcc_0_median', 'mfcc_0_median_min', 'mfcc_0_median_max',
              'mfcc_1_min', 'mfcc_1_max', 'mfcc_1_range', 'mfcc_1_mean',
              'mfcc_1_std', 'mfcc_1_median', 'mfcc_1_median_min', 'mfcc_1_median_max',
              'mfcc_2_min', 'mfcc_2_max', 'mfcc_2_range', 'mfcc_2_mean',
              'mfcc_2_std', 'mfcc_2_median', 'mfcc_2_median_min', 'mfcc_2_median_max',
              'mfcc_3_min', 'mfcc_3_max', 'mfcc_3_range', 'mfcc_3_mean',
              'mfcc_3_std', 'mfcc_3_median', 'mfcc_3_median_min', 'mfcc_3_median_max',
              'mfcc_4_min', 'mfcc_4_max', 'mfcc_4_range', 'mfcc_4_mean',
              'mfcc_4_std', 'mfcc_4_median', 'mfcc_4_median_min', 'mfcc_4_median_max',
              'mfcc_5_min', 'mfcc_5_max', 'mfcc_5_range', 'mfcc_5_mean',
              'mfcc_5_std', 'mfcc_5_median', 'mfcc_5_median_min', 'mfcc_5_median_max',
              'mfcc_6_min', 'mfcc_6_max', 'mfcc_6_range', 'mfcc_6_mean',
              'mfcc_6_std', 'mfcc_6_median', 'mfcc_6_median_min', 'mfcc_6_median_max',
              'mfcc_7_min', 'mfcc_7_max', 'mfcc_7_range', 'mfcc_7_mean',
              'mfcc_7_std', 'mfcc_7_median', 'mfcc_7_median_min', 'mfcc_7_median_max',
              'mfcc_8_min', 'mfcc_8_max', 'mfcc_8_range', 'mfcc_8_mean',
              'mfcc_8_std', 'mfcc_8_median', 'mfcc_8_median_min', 'mfcc_8_median_max',
              'mfcc_9_min', 'mfcc_9_max', 'mfcc_9_range', 'mfcc_9_mean',
              'mfcc_9_std', 'mfcc_9_median', 'mfcc_9_median_min', 'mfcc_9_median_max',
              'mfcc_10_min', 'mfcc_10_max', 'mfcc_10_range', 'mfcc_10_mean',
              'mfcc_10_std', 'mfcc_10_median', 'mfcc_10_median_min', 'mfcc_10_median_max',
              'mfcc_11_min', 'mfcc_11_max', 'mfcc_11_range', 'mfcc_11_mean',
              'mfcc_11_std', 'mfcc_11_median', 'mfcc_11_median_min', 'mfcc_11_median_max',
              'mfcc_12_min', 'mfcc_12_max', 'mfcc_12_range', 'mfcc_12_mean',
              'mfcc_12_std', 'mfcc_12_median', 'mfcc_12_median_min', 'mfcc_12_median_max',
              'mfcc_13_min', 'mfcc_13_max', 'mfcc_13_range', 'mfcc_13_mean',
              'mfcc_13_std', 'mfcc_13_median', 'mfcc_13_median_min', 'mfcc_13_median_max',
              'mfcc_14_min', 'mfcc_14_max', 'mfcc_14_range', 'mfcc_14_mean',
              'mfcc_14_std', 'mfcc_14_median', 'mfcc_14_median_min', 'mfcc_14_median_max',
              'mfcc_15_min', 'mfcc_15_max', 'mfcc_15_range', 'mfcc_15_mean',
              'mfcc_15_std', 'mfcc_15_median', 'mfcc_15_median_min', 'mfcc_15_median_max',
              'mfcc_16_min', 'mfcc_16_max', 'mfcc_16_range', 'mfcc_16_mean',
              'mfcc_16_std', 'mfcc_16_median', 'mfcc_16_median_min', 'mfcc_16_median_max',
              'mfcc_17_min', 'mfcc_17_max', 'mfcc_17_range', 'mfcc_17_mean',
              'mfcc_17_std', 'mfcc_17_median', 'mfcc_17_median_min', 'mfcc_17_median_max',
              'mfcc_18_min', 'mfcc_18_max', 'mfcc_18_range', 'mfcc_18_mean',
              'mfcc_18_std', 'mfcc_18_median', 'mfcc_18_median_min', 'mfcc_18_median_max',
              'mfcc_19_min', 'mfcc_19_max', 'mfcc_19_range', 'mfcc_19_mean',
              'mfcc_19_std', 'mfcc_19_median', 'mfcc_19_median_min', 'mfcc_19_median_max',
              'delta_0_min', 'delta_0_max', 'delta_0_range', 'delta_0_mean',
              'delta_0_std', 'delta_0_median', 'delta_0_median_min', 'delta_0_median_max',
              'delta_1_min', 'delta_1_max', 'delta_1_range', 'delta_1_mean',
              'delta_1_std', 'delta_1_median', 'delta_1_median_min', 'delta_1_median_max',
              'delta_2_min', 'delta_2_max', 'delta_2_range', 'delta_2_mean',
              'delta_2_std', 'delta_2_median', 'delta_2_median_min', 'delta_2_median_max',
              'delta_3_min', 'delta_3_max', 'delta_3_range', 'delta_3_mean',
              'delta_3_std', 'delta_3_median', 'delta_3_median_min', 'delta_3_median_max',
              'delta_4_min', 'delta_4_max', 'delta_4_range', 'delta_4_mean',
              'delta_4_std', 'delta_4_median', 'delta_4_median_min', 'delta_4_median_max',
              'delta_5_min', 'delta_5_max', 'delta_5_range', 'delta_5_mean',
              'delta_5_std', 'delta_5_median', 'delta_5_median_min', 'delta_5_median_max',
              'delta_6_min', 'delta_6_max', 'delta_6_range', 'delta_6_mean',
              'delta_6_std', 'delta_6_median', 'delta_6_median_min', 'delta_6_median_max',
              'delta_7_min', 'delta_7_max', 'delta_7_range', 'delta_7_mean',
              'delta_7_std', 'delta_7_median', 'delta_7_median_min', 'delta_7_median_max',
              'delta_8_min', 'delta_8_max', 'delta_8_range', 'delta_8_mean',
              'delta_8_std', 'delta_8_median', 'delta_8_median_min', 'delta_8_median_max',
              'delta_9_min', 'delta_9_max', 'delta_9_range', 'delta_9_mean',
              'delta_9_std', 'delta_9_median', 'delta_9_median_min', 'delta_9_median_max',
              'delta_10_min', 'delta_10_max', 'delta_10_range', 'delta_10_mean',
              'delta_10_std', 'delta_10_median', 'delta_10_median_min', 'delta_10_median_max',
              'delta_11_min', 'delta_11_max', 'delta_11_range', 'delta_11_mean',
              'delta_11_std', 'delta_11_median', 'delta_11_median_min', 'delta_11_median_max',
              'delta_12_min', 'delta_12_max', 'delta_12_range', 'delta_12_mean',
              'delta_12_std', 'delta_12_median', 'delta_12_median_min', 'delta_12_median_max',
              'delta_13_min', 'delta_13_max', 'delta_13_range', 'delta_13_mean',
              'delta_13_std', 'delta_13_median', 'delta_13_median_min', 'delta_13_median_max',
              'delta_14_min', 'delta_14_max', 'delta_14_range', 'delta_14_mean',
              'delta_14_std', 'delta_14_median', 'delta_14_median_min', 'delta_14_median_max',
              'delta_15_min', 'delta_15_max', 'delta_15_range', 'delta_15_mean',
              'delta_15_std', 'delta_15_median', 'delta_15_median_min', 'delta_15_median_max',
              'delta_16_min', 'delta_16_max', 'delta_16_range', 'delta_16_mean',
              'delta_16_std', 'delta_16_median', 'delta_16_median_min', 'delta_16_median_max',
              'delta_17_min', 'delta_17_max', 'delta_17_range', 'delta_17_mean',
              'delta_17_std', 'delta_17_median', 'delta_17_median_min', 'delta_17_median_max',
              'delta_18_min', 'delta_18_max', 'delta_18_range', 'delta_18_mean',
              'delta_18_std', 'delta_18_median', 'delta_18_median_min', 'delta_18_median_max',
              'delta_19_min', 'delta_19_max', 'delta_19_range', 'delta_19_mean',
              'delta_19_std', 'delta_19_median', 'delta_19_median_min', 'delta_19_median_max',
              ]

df = pd.read_csv('input_features.csv', sep=',', usecols=voice_features)

df_arr = df.to_numpy()
df_labels = pd.read_csv('input_features.csv', sep=',', usecols=['DX'])

## 'DX' is diagnosis labels indicating NC, MCI, and DE. We need to quantify the diagnosis labels.
mask_NC = df_labels['DX'] == 'NC'
df_labels.loc[mask_NC, 'DX'] = 0
mask_MCI = df_labels['DX'] == 'MCI'
df_labels.loc[mask_MCI, 'DX'] = 1
mask_DE = df_labels['DX'] == 'DE'
df_labels.loc[mask_DE, 'DX'] = 2


df_fhs_labels_arr = df_labels.to_numpy()
clf = RandomForestRegressor(n_estimators=100)
clf.fit(df_arr, df_fhs_labels_arr)
importance = clf.feature_importances_
weights_df = pd.DataFrame(columns=["Feature", "Importance", "Importance_norm"])
weights_df['Feature'] = voice_features
weights_df['Importance'] = importance
weights_sum = weights_df['Importance'].sum()
weights_df['Importance_norm'] = weights_df['Importance'] / weights_sum #normalize the importance of features over interval [0, 1]
final_weights_df = weights_df.sort_values(by=['Importance_norm'], ascending=False)
final_weights_df.to_csv("Input_features_importance.csv", sep=',', index=False)