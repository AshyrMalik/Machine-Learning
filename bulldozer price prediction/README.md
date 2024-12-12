# Bluebook Bulldozer Price Regression

## Overview
This project predicts the auction prices of bulldozers using data from past auctions. The objective is to build a robust regression model that can accurately estimate the prices based on various features of the bulldozers.

The project involves:
- Data cleaning and preprocessing.
- Exploratory data analysis (EDA).
- Feature engineering.
- Building and evaluating machine learning models.

## Dataset
The dataset is sourced from the Bluebook Bulldozer Kaggle competition. It contains information about the equipment at the time of sale, including details about make, model, usage, and sale conditions.

## Requirements
To run this project, install the following Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Project Structure
- **Data Cleaning**: Handling missing values and inconsistent data.
- **Feature Engineering**: Extracting useful features from dates, categoricals, and text.
- **Modeling**: Training and evaluating models using algorithms like Random Forest.
- **Evaluation**: Using metrics like Root Mean Square Logarithmic Error (RMSLE) to assess performance.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bluebook-bulldozer-price-regression.git
   ```
2. Navigate to the project directory:
   ```bash
   cd bluebook-bulldozer-price-regression
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook to explore the code and results:
   ```bash
   jupyter notebook end-to-end-bluebook-bulldozer-price-regression.ipynb
   ```

## Key Features
- **Data Preprocessing**: Cleaning and preparing raw data for analysis.
- **Exploratory Data Analysis (EDA)**: Gaining insights into the dataset.
- **Feature Engineering**: Improving model input with derived features.
- **Machine Learning**: Training models using `scikit-learn`'s Random Forest Regressor.

## Results
The final model achieves a competitive RMSLE score on the test set, demonstrating its effectiveness in predicting auction prices for bulldozers.

## Future Work
- Integrate advanced algorithms like Gradient Boosting or XGBoost for improved performance.
- Explore hyperparameter optimization techniques such as GridSearchCV or Optuna.
- Deploy the model using Flask or FastAPI for real-world applications.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for enhancements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
