# DocAssist: Intelligent Medical Decision Support System

<div align="center">
  <img src="visuals/docassist_logo.png" alt="DocAssist AI Logo" width="200"/>
  <br>
  <p><i>Empowering Healthcare Decisions with AI</i></p>
</div>

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/realranjan/DOCASSIST-AI)
[![Live Demo](https://img.shields.io/badge/Live-Demo-green.svg)](https://docassist-ai-alpha.vercel.app)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Problem Statement
In the current medical landscape, doctors face significant challenges in processing vast amounts of patient data to make treatment decisions. This can lead to:
- Delays in treatment decisions
- Inconsistencies in recommendations
- Suboptimal treatment choices
- Increased cognitive load on healthcare providers

DocAssist addresses these challenges by providing an AI-powered decision support tool that assists healthcare professionals in making informed, data-driven clinical decisions.

## System Architecture
<div align="center">
  <img src="visuals/system_architecture.png" alt="DocAssist System Architecture" width="800"/>
  <br>
  <p><i>DocAssist AI System Architecture: Integrating Healthcare Support, Data Analysis, and Personalized Recommendations</i></p>
</div>

## Web Interface
The system features a modern, intuitive web interface built with:
- **Frontend**: Next.js, Tailwind CSS, shadcn/ui components
- **Backend**: Flask API server
- **Deployment**: Vercel (Frontend), Render (Backend)

<div align="center">
  <img src="ui visuals/dashboard.png" alt="DocAssist Web Interface" width="800"/>
  <br>
  <p><i>DocAssist AI Web Interface: Modern, Intuitive, and User-Friendly Design</i></p>
</div>

### Key UI Features
- 📊 Interactive dashboards for data visualization
- 📱 Responsive design for all devices
- 🔄 Real-time analysis updates
- 📄 PDF report generation and viewing
- 🎨 Modern and clean user interface
- 🔒 Secure data handling

## Dataset Information
The dataset is sourced from a private hospital in Indonesia and contains comprehensive patient laboratory test results used for treatment recommendations.

### Features
| Feature Name  | Data Type    | Description |
|--------------|--------------|-------------|
| HAEMATOCRIT  | Continuous   | Proportion of blood volume occupied by red blood cells |
| HAEMOGLOBINS | Continuous   | Oxygen-carrying protein in red blood cells |
| ERYTHROCYTE  | Continuous   | Red blood cell count per volume |
| LEUCOCYTE    | Continuous   | White blood cell count per volume |
| THROMBOCYTE  | Continuous   | Platelet count per volume |
| MCH          | Continuous   | Mean Corpuscular Hemoglobin |
| MCHC         | Continuous   | Mean Corpuscular Hemoglobin Concentration |
| MCV          | Continuous   | Mean Corpuscular Volume |
| AGE          | Continuous   | Patient age |
| SEX          | Nominal      | Patient gender (M/F) |
| SOURCE       | Nominal      | Patient care type (1 = In-care, 0 = Out-care) |

## Project Structure
```
DOCASSIST-AI/
├── backend/                 # Flask API server
│   ├── app.py              # Main application
│   ├── models/             # ML models
│   └── utils/              # Helper functions
├── frontend/               # Next.js frontend
│   ├── components/         # UI components
│   ├── pages/             # Application pages
│   └── styles/            # CSS styles
├── data/                  # Dataset files
├── notebooks/             # Analysis notebooks
├── ui visuals/            # Interface mockups
└── visuals/              # Project diagrams
```

## Technical Implementation

### Data Preprocessing
1. **Data Cleaning**
   - Handling missing values
   - Removing duplicate entries
   - Outlier detection and treatment

2. **Feature Engineering**
   - Creation of derived features (e.g., thrombocyte-leucocyte ratio)
   - Encoding of categorical variables
   - Scaling numerical features using RobustScaler

3. **Exploratory Data Analysis**
   - Distribution analysis of class labels
   - Gender and age demographics
   - Feature correlation analysis
   - Statistical visualization of numerical features

### Model Performance

#### Pre-tuning Performance
| Model                  | Train Accuracy | Test Accuracy | ROC AUC | Precision |
|-----------------------|----------------|---------------|---------|-----------|
| Random Forest         | 100.00%        | 75.88%       | 0.80    | 0.74      |
| CatBoost             | 87.19%         | 75.31%       | 0.82    | 0.73      |
| LightGBM             | 92.94%         | 74.52%       | 0.81    | 0.71      |
| XGBoost              | 98.67%         | 74.41%       | 0.81    | 0.70      |
| AdaBoost             | 75.57%         | 74.07%       | 0.79    | 0.72      |
| Support Vector Machine| 76.79%         | 73.61%       | 0.79    | 0.74      |
| K-Nearest Neighbors   | 80.31%         | 72.03%       | 0.75    | 0.67      |
| Logistic Regression   | 72.83%         | 71.46%       | 0.75    | 0.70      |

#### Post-tuning Performance
| Model              | Train Accuracy | Test Accuracy | ROC AUC | Precision |
|-------------------|----------------|---------------|---------|-----------|
| Tuned XGBoost     | 96.57%        | 77.12%       | 0.81    | 0.76      |
| Tuned Random Forest| 91.67%        | 76.67%       | 0.81    | 0.76      |
| Tuned CatBoost    | 91.47%        | 76.67%       | 0.81    | 0.74      |
| Tuned LightGBM    | 88.18%        | 77.34%       | 0.82    | 0.75      |

### Key Findings
- **Optimal Model Selection**: LightGBM achieved the highest test accuracy (77.34%) after tuning
- **Reduced Overfitting**: Training accuracy decreased while test accuracy increased
- **Consistent Performance**: All tuned models showed ROC AUC scores of 0.81
- **High Precision**: XGBoost and Random Forest achieved 0.76 precision after tuning

### Final Model: LightGBM
After comprehensive evaluation and hyperparameter tuning, **LightGBM** was selected as the final production model for the following reasons:

#### Performance Metrics
- **Test Accuracy**: 77.34% (highest among all models)
- **Train Accuracy**: 88.18% (good balance between bias and variance)
- **ROC AUC**: 0.82 (strong classification capability)
- **Precision**: 0.75 (reliable positive predictions)

#### Key Advantages
- **Gradient Boosting Framework**: LightGBM uses a highly efficient gradient boosting framework
- **Leaf-wise Growth**: Employs leaf-wise tree growth strategy for better accuracy
- **Memory Efficient**: Uses histogram-based algorithms to handle categorical features
- **Fast Training**: Significantly faster training speed compared to traditional GBDT
- **Handling Imbalanced Data**: Better performance on slightly imbalanced medical datasets

#### Model Configuration
```python
lightgbm_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}
```

#### Production Implementation
The model is deployed with:
- Regular retraining pipeline for maintaining accuracy
- Model versioning for tracking performance
- Monitoring system for detecting drift
- Fallback mechanisms for reliable predictions

### Features Available in Demo
- ✅ Blood test report analysis
- ✅ Real-time parameter visualization
- ✅ PDF report generation
- ✅ Treatment recommendations
- ✅ Historical data tracking

## Installation and Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Installation
```bash
# Clone the repository
git clone [repository-url]

# Navigate to project directory
cd docassist

# Install dependencies
pip install -r requirements.txt
```

### Usage
1. Data Preparation:
```python
# Import required libraries
import pandas as pd
from docassist import preprocess

# Load and preprocess data
data = pd.read_csv('path_to_data.csv')
processed_data = preprocess.prepare_data(data)
```

2. Model Training:
```python
# Import model trainer
from docassist import model

# Train model
trained_model = model.train(processed_data)
```

3. Making Predictions:
```python
# Get predictions
predictions = model.predict(patient_data)
```

## Future Improvements
1. **Data Enhancement**
   - Expand dataset diversity
   - Include additional medical parameters
   - Incorporate temporal patient data

2. **Technical Improvements**
   - Implement deep learning models
   - Develop REST API for model serving
   - Create web-based user interface
   - Add real-time monitoring capabilities

3. **Clinical Integration**
   - Integrate with Electronic Health Records (EHR)
   - Implement HIPAA compliance measures
   - Add support for multiple medical specialties

## Contributing
We welcome contributions to improve DocAssist. Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
- **Ranjan Vernekar** - Project Lead
  - LinkedIn: [Ranjan Vernekar](https://www.linkedin.com/in/ranjan-vernekar-a93b08252/)
  - GitHub: [@realranjan](https://github.com/realranjan)

## Acknowledgments
- Private hospital in Indonesia for providing the dataset
- Healthcare professionals who provided domain expertise
- Open-source community for machine learning tools and libraries

---
<div align="center">
  <p>Made with ❤️ by the DocAssist AI Team</p>
  <p>© 2024 DocAssist AI. All rights reserved.</p>
</div> 
