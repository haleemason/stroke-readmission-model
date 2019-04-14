# stroke-readmission-model
Capstone project for Data Science Masters program at the University of Wisconsin.  

## Overview
**Contributors:** Samuel Edeh and Halee Mason <br>
**Project:** Predicting Risk of Re-Admission for Stroke Patients <br>
**Completion Date:** August 2017 <br> 
**Data Source:** [Nationwide Re-admissions Database (NRD)](https://www.hcup-us.ahrq.gov/nrdoverview.jsp)

## Description:
Built a predictive model using TensorFlow that was optimized to forecast the risk of rehospitalization in the year following stroke. Model was trained on data from the [Nationwide Re-admissions Database (NRD)](https://www.hcup-us.ahrq.gov/nrdoverview.jsp) from the Healthcare Cost and Utilization Project (HCUP) State Inpatient Databases (SID). Deployed the model using the Google Cloud Platform.

## Motivation:
Collaborated with the client to understand the business value that successful data science application would have for this project and focused on model explainability by using techniques such as [LIME](https://github.com/marcotcr/lime). Implemented model to drive patient outcomes to help the client segment high risk patients and engage them more closely. The final deliverable was an API endpoint which allowed for the client to input patient specific parameters (input features) and for the model to output a personalized risk score of rehospitalization. 

