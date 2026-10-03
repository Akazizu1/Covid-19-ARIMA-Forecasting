# COVID 19 Data Analysis and ARIMA Forecasting

This project started as a Data Science lab project after I came across an Instagram post about ARIMA and time series forecasting. I wanted to understand how ARIMA worked in practice, so I decided to apply it to a realistic dataset instead of using a small example.

I chose COVID 19 data because it contained information from many countries over time and provided a good opportunity to explore both data analysis and time series forecasting.

## Project Overview

The dataset contains COVID 19 information from **187 countries**, covering the period from **January 22, 2020 to July 27, 2020**.

The project covers:

1. Data cleaning and preprocessing
2. Exploratory data analysis
3. Country and WHO region comparisons
4. COVID 19 trend visualization
5. Death rate analysis
6. ARIMA time series forecasting
7. Model evaluation using MAE, RMSE and MAPE
8. 30 day forecasting

## Dataset

The dataset contains **35,156 records** and 10 columns.

The main columns include:

* Date
* Country/Region
* Confirmed
* Deaths
* Recovered
* Active
* New cases
* New deaths
* New recovered
* WHO Region

The data covers 187 countries and six WHO regions.

## Data Cleaning

Before starting the analysis, I performed some basic cleaning steps:

* Converted the Date column into a proper date format
* Removed duplicate records
* Removed rows with missing values
* Removed extra spaces from country names
* Handled small negative values caused by data corrections by setting them to zero
* Sorted the data by country and date

## Exploratory Data Analysis

The project includes several visualizations to understand the dataset.

These include:

* Top 10 countries by confirmed cases, deaths, recoveries or active cases
* Worldwide COVID 19 trends over time
* COVID 19 cases by WHO region
* Countries with the highest death rates
* Optional comparison of multiple countries

The analysis can be performed on confirmed cases, deaths, recovered cases or active cases.

## ARIMA Forecasting

For the forecasting part, I used the ARIMA model.

The program allows the user to select either a specific country or the worldwide totals for forecasting.

Before building the model, I used the Augmented Dickey Fuller test to check whether the time series was stationary. Differencing was then applied when required.

I tested small combinations of ARIMA parameters and selected the combination with the lowest AIC value.

For the worldwide confirmed cases data, the selected model was:

**ARIMA (2, 2, 2)**

## Model Evaluation

Instead of evaluating the model on the same data it was trained on, the time series is divided into training and testing data.

The model is trained on the first 80 percent of the data and tested on the remaining 20 percent.

The worldwide confirmed cases model produced the following results:

| Metric |     Result |
| ------ | ---------: |
| ARIMA  |  (2, 2, 2) |
| MAE    | 503,443.91 |
| RMSE   | 711,858.13 |
| MAPE   |      3.52% |

I focused mainly on MAPE because it is easier to interpret for this dataset. Since the worldwide cumulative case numbers are in the millions, the MAE can look very large when viewed by itself.

These results should be considered in the context of the historical COVID 19 dataset and should not be interpreted as a current COVID 19 prediction.

## 30 Day Forecast

After evaluating the model, the selected ARIMA model is trained again using the complete available dataset.

The final model is then used to generate a 30 day forecast.

The forecast also includes a confidence interval to show the uncertainty around the predicted values.

## Challenges and Learning

One of the main challenges was understanding how to properly evaluate a time series model.

Initially, I compared the model's fitted values with the same data used to train it. This gave very low error values, but I learned that this does not properly test how the model performs on unseen data.

I changed the project to use a training and testing split so that the model could be evaluated on data it had not seen during training.

Other challenges included handling data corrections, choosing the differencing order and finding suitable ARIMA parameters.

Working through these issues helped me understand that Data Science is not only about getting a model to run. Understanding the data and making sure the results are being interpreted correctly is just as important.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Statsmodels

## Project Structure

```text
Covid 19
│
├── full_grouped.csv
├── ARIMA.py
└── README.md
```

## How to Run

Clone the repository and install the required libraries:

```bash
pip install pandas numpy matplotlib statsmodels
```

Run the project with:

```bash
python ARIMA.py
```

The program will ask you to select the column you want to analyze and then allow you to choose a country or the worldwide data for forecasting.

## Conclusion

This project gave me a practical introduction to time series forecasting using ARIMA. It also helped me practice the complete process of working with a real dataset, from cleaning and exploration to visualization, model building and evaluation.

What started as a Data Science lab assignment and an interest in something I saw on Instagram became a useful project for understanding how time series forecasting works in practice.
