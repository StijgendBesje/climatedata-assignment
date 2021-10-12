# climatedata-assignment
Assignment analyzing climate weather data

## Problem Introduction
Predicting max temperature for tomorrow using one year of past weather data, in Seattle. This is a supervised, regerssion machine learning problem. The features are data for the city, and the target is the temperature. 

## Dataset 
Data is available on the National centers for environmental information. Other datasets can be found in [this link](https://www.ncdc.noaa.gov/cdo-web/datasets)

## Roadmap 
This repository follows the following steps: 
1. State the question and required data
2. Acquire data 
3. Identify and preprocess data: missing data, anomalies
4. Prepare data for modelling
5. Establish a baseline model 
6. Train model on trainig data
7. Make predictions on test data
8. Compare predictions and calculate performance metrics 
9. If performance not satisfactory: adjust parameters, acquire more data or use different modeling techniques 
10. Interpret model and report results

## Run package

create a virtual environment 

`sudo apt-get install graphviz`
`mkdir env data/anomoly_detection data/hist data/heatmap data/boxplot`
`python{version} -m venv env`
`pip install -r requirements/base.txt`