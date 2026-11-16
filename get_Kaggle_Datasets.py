import kagglehub
from kagglehub import KaggleDatasetAdapter

def get_kaggle(s_data_file, s_file_path):
# Noisy Medical Document Images – Hospital Bills & Discharge Summaries
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  s_data_file,
  s_file_path,
)
return df

# Noisy Medical Document Images – Hospital Bills & Discharge Summaries
s_data_file ="devp1866/noisy-medical-document-images-ocr"

# Polycystic Ovary Syndrome PCOS
s_data_file ="lucass0s0/polycystic-ovary-syndrome-pcos"

# Surgical Scene Segmentation in Robotic Gastrectomy
s_data_file = "yjh4374/sisvse-dataset"

# Chest X-ray Dataset for Tuberculosis Segmentation
# Chest X-ray Organized Lung Segmentation Masks
s_data_file = "iamtapendu/chest-x-ray-lungs-segmentation"


# Chronic Kidney Disease Dataset
# Comprehensive Health Data for Chronic Kidney Disease Analysis
s_data_file = "rabieelkharoua/chronic-kidney-disease-dataset-analysis"


# DenialIQ: 120K Medical Claims | X12 Denial Codes
# 120K Synthetic Claims with X12 Denial Codes, Appeal Labels & Payer Rules for RCM
s_data_file = "nudratabbas/denialiq-120k-medical-claims-x12-denial-codes"


# Tuberculosis X-Ray Dataset (Synthetic)
# Synthetic Data for Tuberculosis Detection and Classification
s_data_file = "miadul/tuberculosis-x-ray-dataset-synthetic"


# Kidney Disease Risk Dataset
# Chronic Kidney Disease (C	KD) and Dialysis Risk Prediction
s_data_file = "miadul/kidney-disease-risk-dataset"


# Patient Records 100K patients · 15 conditions
# 100K patients · 15 conditions · ICD-10 · lab results · medications · 30-day read
s_data_file = "sergionefedov/patient-records-100k-patients-15-conditions"


# Medical Device Manuals Global Dataset 2025
# Device manual with approvals, uses, cautions & specs for ML/NLP analysis in 2025
s_data_file = "pratyushpuri/global-medical-device-manuals-dataset-2025"


# Brain CT Medical Imaging Colorized Dataset
# Brain, CT, MRI, X-ray
s_data_file = "shuvokumarbasakbd/brain-ct-medical-imaging-colorized-dataset"


# Pharmaceutical Drugs and Vitamins Synthetic Images
# Synthetic Images of Common Pharmaceutical Drugs found in the Philippines
s_data_file = "vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images"


# Transformer Models for PII Detection
# https://www.kaggle.com/code/gpreda/transformer-models-for-pii-detection"


# NBME - Score Clinical Patient Notes
# Identify Key Phrases in Patient Notes from Medical Licensing Exams
# https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/305793"


# NBME - Score Clinical Patient Notes
# Identify Key Phrases in Patient Notes from Medical Licensing Exams
# https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/discussion/305793
s_data_file = "nbme-score-clinical-patient-notes"

