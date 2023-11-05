# Website conversion prediction | Midterm Project ML-Zoomcamp

This repository contains the code and materials for the midterm project of the Zoomcamp Machine Learning course. The project focuses on predicting whether visitors of stuffmart.com will convert into paying customers given several attributes collected on these users.

## Overview

In this project, I have worked on a [Kaggle binary classification dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-conversion-dataset-for-stuffmart-com) that contains information related to potential leads and their interactions with a hypothetical business or website. The purpose of the current project is to predict wether these leads will convert into paying customers based on their previous interactions. 

I have choosen this particular problem because I am keen on applying machine learning on the digital marketing space and I can see such a model being relevant for a website operator who wants to maximize their conversion rate by for example targeting campaigns at users who can be very close to convert to help them with that extra push.

After testing with several models XGBoost was choosen based on the AUC ROC curve criteria. You can check my exploration journey on this [notebook](https://github.com/gdumie01/mlzoomcamp-midterm-conv-prediction/blob/main/notebook.ipynb).

## Dataset

### Where to find

Data is available in the [data folder](https://github.com/gdumie01/mlzoomcamp-midterm-conv-prediction/tree/main/data) of this project, but you can also download it from [Kaggle](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-conversion-dataset-for-stuffmart-com).

### Data explanation

The dataset includes attributes such as age, gender, location (with a focus on major cities in Pakistan), lead source, engagement metrics, lead status, email interactions, device type, referral sources, form submissions, downloads, click-through rates, response times, follow-up emails, social media engagement, and payment history. 

The dataset also includes a binary target variable, "Conversion," which indicates whether a lead has converted (1) or not (0) based on strict criteria. Leads are marked as converted if they meet specific conditions related to lead engagement, behavior, age, and location.

The dataset was already splitted into training and testing but in my approach I've opted to combine them both to follow the methodology teached in ML Zoomcamp.

## How to Run the Code

### Dependencies
For the notebook you´ll need:
- Python 3.9+
- NumPy
- Pandas
- Scikit-Learn
- XGBoost
- Seaborn

To run the python scripts you´ll need:
- Flask
- gunicorn

Project contains a `Pipfile` and a `Pipfile.lock` to setup the virtual environment with the proper versions for train and predict scripts.

Finally, you´ll need to have Docker installed on your machine to build and execute the docker image that will serve the prediction webservice.

### Setup instructions

#### 1. Clone this repository.
```
git clone https://github.com/gdumie01/mlzoomcamp-midterm-conv-prediction.git
```
#### 2. Build the docker image
```
docker build -t {build-tag} .
```
`build-tag`: Specifies any user-defined tag for docker image. eg. `conv-prediction`

#### 3. Run the docker image

```
docker run -it -p 9696:9696 {build-tag}:latest
```
in case you used the suggested tag above code would be:
```
docker run -it -p 9696:9696 conv-prediction:latest
```

#### 4. Now use the predict-tester script to test the service
In another terminal run the predict-tester.py.
```
python predict-tester.py
```
6. If you want to use other data you can just change it in the scritp and run again
## Sample output
![Sample output](https://github.com/gdumie01/mlzoomcamp-midterm-conv-prediction/blob/main/data/sample_output.png)
