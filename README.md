# Data Preparation and Visualization_Group 7
# **US Accidents - Exploratory Data Analysis & Prediction**

## **Introduction**
Traffic accidents have always been one of the leading public safety issues in the United States as well as around the world. Understanding the factors that influence accidents (such as weather conditions, transportation infrastructure, and time of day) not only helps improve traffic planning but also supports authorities in implementing effective preventive measures.

This project focuses on Exploratory Data Analysis (EDA) and Data Preparation using a U.S. traffic accident dataset, with the aim of uncovering important insights and building a foundation for forecasting.

## **Dataset Description**
The dataset used in this project is **US Accidents (2020 - 2023)**, one of the most up-to-date and detailed nationwide traffic accident datasets.
* **Data source:** Collected from various traffic APIs (MapQuest, Bing, etc.).
* **Scope:** Includes over 3 million accident records across 49 U.S. states.
* **Attributes:** The dataset contains detailed information such as:
    * **Severity:** A scale from 1 to 4 indicating the impact level of the accident.
    * **Time & Location:** Start/end times, latitude/longitude, street names, city, and state.
    * **Environmental conditions:** Temperature, humidity, visibility, wind speed, and weather conditions (rain, snow, fog, etc.).
    * **Infrastructure features:** Presence of traffic lights, signs, speed bumps, and more.

## **Project Objectives**
Based on the requirements of the **Data Preparation and Visualization** course, the main objectives are as follows:

1.  **Data Cleaning & Preprocessing:** Handle noisy data, manage missing values, and standardize data formats.
2.  **Exploratory Data Analysis (EDA):**
    * Analyze the distribution of accidents over time (hour, day, month) and across locations (states, cities).
    * Evaluate the impact of weather conditions and traffic infrastructure on accident severity.
    * Visualize the data to tell a story (Data Storytelling) about the traffic landscape in the U.S.
3.  **Feature Engineering & Selection:** Create useful new features and remove multicollinear variables (using VIF).
4.  **Modeling Strategy:** Experiment with machine learning models (such as Random Forest, XGBoost, LightGBM, CatBoost) for prediction and performance evaluation.