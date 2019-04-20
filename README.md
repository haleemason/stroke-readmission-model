# stroke-readmission-model
Capstone project for Data Science Masters program at the University of Wisconsin.  

## Overview
**Contributors:** Samuel Edeh and Halee Mason <br>
**Capstone:** Predicting Risk of Re-Admission for Stroke Patients <br>
**Completion Date:** August 2017 <br> 
**Data Source:** [Nationwide Re-admissions Database (NRD)](https://www.hcup-us.ahrq.gov/nrdoverview.jsp)

## Description
Built a predictive model using TensorFlow that was optimized to forecast the risk of rehospitalization in the year following stroke. Model was trained on data from the [Nationwide Re-admissions Database (NRD)](https://www.hcup-us.ahrq.gov/nrdoverview.jsp) from the Healthcare Cost and Utilization Project (HCUP) State Inpatient Databases (SID). Deployed the model using the Google Cloud Platform [here](https://capstoneproj.herokuapp.com/).

## Motivation
Collaborated with the client to understand the business value that successful data science application would have for this project and focused on model explainability by using techniques such as [LIME](https://github.com/marcotcr/lime). Implemented model to drive patient outcomes to help the client segment high risk patients and engage them more closely. The final deliverable was an API endpoint which allowed for the client to input patient specific parameters (input features) and for the model to output a personalized risk score of rehospitalization. 


## Demo
`call_cloud_service.ipynb` provides a demonstration of how to connect to the Google Cloud ML Service to make a prediction from a trained model stored in Google Cloud.

## Accessing the App
The app is live [here](https://capstoneproj.herokuapp.com/).

Follow the text prompt in the fields and enter information correctly. For e.g., when the prompt asks you to enter "y/n", you could enter "`y`" or "`yes`", and it will consider it as "`yes`" that the patient has the condition. Entering "`n`" or "`no`" indicates that the patient does not have the condition. 

A patient will have 13 input features. All are yes or no features except the 8th, with is the patients age. Example: `patient_1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86, 0.0, 0.0, 0.0, 0.0, 1.0]` correspond to:

1. pay1_private
2. metro
3. diabetes
4. copd
5. ckd
6. chf
7. atrial_fib
8. age
9. hyperlipidemia
10. sex
11. nicotine
12. obesity
13. hypertension