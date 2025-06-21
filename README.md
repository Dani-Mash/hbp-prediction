# Early Detection of Hypertensive Disorders in Pregnancy: A Machine Learning Approach

A comprehensive machine learning solution for early detection of hypertensive disorders in pregnancy using multimodal clinical data. This project demonstrates advanced data science techniques including feature engineering, text processing, and budget-constrained model evaluation.

## Project Overview

**Business Objective**: Develop a predictive model to identify pregnant women who should be referred for additional (expensive) blood testing at week 15 of gestation, maximizing true-case detection under a fixed testing budget.

**Key Achievements**:
- XGBoost model achieves 58.1% recall and 83.3% precision (AUC: 0.974)
- Budget optimization: Top 100 patients capture 70.9% of true cases
- Multimodal data processing (structured clinical data + unstructured text)
- Advanced feature engineering with 184 engineered features

## Installation & Setup

### Prerequisites
- Python 3.8 or higher

### Dependencies Installation

```bash
# Core data science packages
pip install pandas numpy matplotlib seaborn scikit-learn scipy

# Machine learning
pip install xgboost

# Alternative: Install all at once
pip install -r requirements.txt
```

### XGBoost Installation (macOS)
```bash
# If XGBoost installation fails on macOS
brew install libomp
pip install xgboost
```

## Project Structure

```
Maccabi_Home_Task/
├── README.md                           # This documentation
├── requirements.txt                    # Python dependencies
├── ds_maccabi_pipline.ipynb           # Main analysis notebook
├── ds_assignment_data.csv             # Raw clinical dataset
├── X_y_combined.csv                   # Processed features + target
└── Clinical_Data_Analysis_Report.pdf   # Comprehensive analysis report
```

## Data Requirements

### Input Dataset
- **File**: `ds_assignment_data.csv` - raw clinical data
- **Size**: 10,000+ patient records with ~4% positive cases
- **Format**: CSV with 157 columns including:
  - Demographics: age, socioeconomic status
  - Laboratory results: CBC, biochemical markers, urine analysis
  - Blood pressure measurements: systolic/diastolic trends
  - Diagnosis history: ICD-9 codes from 4 and 24-month windows
  - Clinical text: physician notes and documentation
  - Target variable: Y (0/1 for hypertensive complications)

### Data Quality
- Expected class imbalance: 95.68% negative, 4.32% positive cases
- Missing data handled through strategic imputation
- Data leakage prevention: excluded post-week-15 features

**Feature Engineering Strategy:**
1. **Missing Data Handling**: Strategic imputation with missing indicators
2. **Feature Creation**: Aggregation of diagnosis history, blood pressure trends, lab ratios
3. **Text Processing**: TF-IDF vectorization with SVD dimensionality reduction
4. **Data Leakage Prevention**: Exclusion of post-week-15 and outcome-derived features

**Model Architecture:**
- **Primary Algorithm**: XGBoost with class imbalance handling
- **Evaluation**: Stratified 80/20 train-test split
- **Metrics**: Recall and precision focus for clinical relevance

## Running the Analysis

### Step-by-Step Execution

1. **Launch Jupyter Environment:**
```bash
jupyter notebook
```

2. **Open and Execute**:
   - Navigate to `ds_maccabi_pipline.ipynb`
   - Run cells sequentially from top to bottom

3. **Execution Flow**:
   - **Data Loading & Exploration** (Cells 1-8): Dataset overview, missingness analysis
   - **Feature Engineering** (Cells 9-15): Preprocessing, aggregation, text features
   - **Modeling** (Cells 16-20): Train-test split, model training, evaluation
   - **Results Analysis** (Cells 21-25): Confusion matrices, feature importance

**Model Performance Metrics:**
- XGBoost: 58.1% recall, 83.3% precision, AUC 0.974
- Random Forest: 2.3% recall, 100% precision, AUC 0.85

## Key Outputs

### Model Performance
- **XGBoost**: 58.1% recall, 83.3% precision, AUC 0.974
- **Random Forest**: 2.3% recall, 100% precision, AUC 0.85
- **Confusion Matrices**: Detailed classification results

### Budget-Constrained Analysis
| Top-K Patients | Precision@K | Recall@K | Efficiency Gain |
|----------------|-------------|----------|-----------------|
| 50 | 88% | 51.2% | 20.5x |
| 100 | 61% | 70.9% | 14.2x |
| 200 | 38.5% | 89.5% | 9.0x |

### Feature Importance
- Blood pressure trends and ranges
- Diagnosis history recency
- Laboratory ratios (neutrophil-to-lymphocyte)
- Clinical flags and demographic factors
- Text-derived semantic features

## Analysis Highlights

### Clinical Insights
Our comprehensive analysis revealed several clinically significant findings:

- **Age Effect**: Patients with complications had significantly higher mean age (30.8 vs 29.7 years, p<0.001), confirming maternal age as a risk factor
- **Socioeconomic Factors**: Weak but statistically significant association between capitation coefficient and outcomes (p<0.001)
- **Blood Pressure Trends**: Early warning signals detected before clinical thresholds, suggesting subclinical changes
- **Diagnosis History**: Recent diagnoses (within 4-24 months) strongly correlate with elevated risk
- **Laboratory Patterns**: Neutrophil-to-lymphocyte ratio and hemoglobin levels provide additional predictive signals

### Technical Achievements
The analysis demonstrates advanced data science capabilities:

- **Multimodal Integration**: Successfully combined structured clinical data with unstructured Hebrew text
- **Feature Engineering**: Created 184 engineered features from 157 raw columns through sophisticated aggregation
- **Class Imbalance Handling**: Achieved robust performance despite 4.32% positive rate using strategic techniques
- **Text Mining**: Extracted meaningful features from clinical notes using TF-IDF and SVD dimensionality reduction


### Business Impact
The model delivers substantial value for healthcare systems:

- **Cost Reduction**: 14.2x efficiency gain over random selection for top 100 patients
- **Early Detection**: Identifies high-risk patients before clinical symptoms appear
- **Resource Optimization**: Enables targeted allocation of expensive laboratory testing
- **Quality Improvement**: Systematic, evidence-based approach to risk assessment
- **Scalability**: Framework applicable to other clinical prediction tasks

### Model Performance
- **XGBoost**: 58.1% recall, 83.3% precision, AUC 0.974 - optimal for clinical deployment
- **Random Forest**: 2.3% recall, 100% precision - demonstrates class imbalance challenges
- **Budget Efficiency**: Top 100 patients capture 70.9% of true cases with 61% precision
- **Clinical Relevance**: Focus on recall ensures minimal missed cases, critical for patient safety


