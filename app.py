from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        employee_data_file = request.files['employee_data']
        engagement_data_file = request.files['engagement_data']
        
        if employee_data_file and engagement_data_file:
            employee_data_df = pd.read_csv(employee_data_file)
            employee_engagement_survey_data_df = pd.read_csv(engagement_data_file)

            # Rename EmpID to Employee ID for future merging data set purpose
            print ("\nRenaming EmpID to EmployeeID.....\n")
            employee_data_df = employee_data_df.rename(columns={'EmpID' : 'Employee ID'})
            print (employee_data_df.head())

            print ("\nDrop unwanted column in dataset 1.....\n")
            column_to_drop = ['FirstName','LastName','StartDate','ExitDate','Title','Supervisor','ADEmail','BusinessUnit','EmployeeClassificationType','EmployeeStatus','TerminationType','TerminationDescription','Division','DOB','State','JobFunctionDescription','LocationCode']
            employee_data_df = employee_data_df.drop(columns = column_to_drop)
            print (employee_data_df.dtypes)

            print ("\nDrop unwanted column in dataset 2.....\n")
            columns_to_drop = ['Survey Date']
            employee_engagement_survey_data_df = employee_engagement_survey_data_df.drop(columns = columns_to_drop)
            print (employee_engagement_survey_data_df.dtypes)

            # Preprocess employee data
            preprocessed_employee_data = preprocess(employee_data_df)
            # Preprocess engagement data
            preprocessed_engagement_data = preprocess(employee_engagement_survey_data_df)

            # Transfroming categorical data into numerical data
            print("\nTransforming dataset.....\n")
            columns_to_encode = ['PayZone','GenderCode','RaceDesc','MaritalDesc','Performance Score','EmployeeType','DepartmentType']
            label_encoders = {}
            def encode_and_insert(df, column_name):
                label_encoder = LabelEncoder()
                encoded_values = label_encoder.fit_transform(df[column_name])
                encoded_column_name = f'{column_name}_Encoded'
                df.insert(loc=df.columns.get_loc(column_name) + 1, column=encoded_column_name, value=encoded_values)
                label_encoders[column_name] = label_encoder
            for col in columns_to_encode:
                encode_and_insert(preprocessed_employee_data, col)
            print(preprocessed_employee_data.head(10))

            preprocessed_employee_data.to_csv(f'Cleaned Employee DataSet.csv', index=False)
            preprocessed_engagement_data.to_csv(f'Cleaned Engagement Survery DataSet.csv', index=False)

            # Merge datasets
            merged_data = merge_datasets(preprocessed_employee_data, preprocessed_engagement_data)
            
            # Save merged dataset
            merged_data.to_csv('Merged_Data.csv', index=False)

            # Load merged dataset
            merged_data = pd.read_csv('Merged_Data.csv')

            # Train Model
            model = train_model(merged_data)
            
            # Save Model
            joblib.dump((model, label_encoders), 'model_and_encoders.pkl')
            
            return redirect(url_for('prediction_input'))
        else:
            return redirect(url_for('index'))


def preprocess(df: pd.DataFrame):
    print ("\nSort the datafrane with EmployeeID in ascending order.....\n")
    df = df.sort_values(by='Employee ID')

    # Handle missing values
    # For simplicity, let's fill missing values with zeros for numerical columns
    df = df.fillna(0)

    return df

def merge_datasets(employee_data, engagement_data):
    return pd.merge(employee_data, engagement_data, on='Employee ID')

def train_model(merged_df: pd.DataFrame):
    
    # Separate the features and the target variable
    number_columns = merged_df.select_dtypes(include=['int64', 'float64']).drop('Performance Score_Encoded', axis=1)
    X = number_columns.drop(columns='Employee ID')
    y = merged_df['Performance Score_Encoded']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply random over-sampling to the training data
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    # Standardizing the numerical features
    scaler = StandardScaler()

    # Select numerical columns to scale
    numerical_cols = X_train_resampled.select_dtypes(include=['int64', 'float64']).columns
    X_train_resampled[numerical_cols] = scaler.fit_transform(X_train_resampled[numerical_cols])

    # Scale numerical features of test data
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # ########################################################
    # # Random Forest Classifier
    # # Define the parameter grid for hyperparameter tuning
    # param_grid_rf = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [None, 10, 20],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2]
    # }

    # # Initialize the Random Forest model
    # model_rf = RandomForestClassifier(random_state=42)

    # # Initialize GridSearchCV
    # grid_search_rf = GridSearchCV(estimator=model_rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, scoring='accuracy')

    # # Fit GridSearchCV on the training data
    # grid_search_rf.fit(X_train_resampled, y_train_resampled)

    # # Print the best hyperparameters
    # print(f"Best hyperparameters for Random Forest: {grid_search_rf.best_params_}")

    # # Get the best model from grid search
    # best_rf = grid_search_rf.best_estimator_

    # # Perform cross-validation with the best model
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # scores_rf = cross_val_score(best_rf, X_train_resampled, y_train_resampled, cv=kf)

    # print(f"Random Forest CV scores: {scores_rf}")
    # print(f"Mean Random Forest CV score: {scores_rf.mean():.4f}")

    # # Fit the best model on the training data
    # best_rf.fit(X_train_resampled, y_train_resampled)

    # # Predict on the test data
    # y_pred_rf = best_rf.predict(X_test)

    # # Evaluate the model
    # accuracy_rf = accuracy_score(y_test, y_pred_rf)
    # print(f"Accuracy: {accuracy_rf:.4f}")

    # print('Classification Report:')
    # print(classification_report(y_test, y_pred_rf))

    # print('Confusion Matrix:')
    # print(confusion_matrix(y_test, y_pred_rf))

    ########################################################
    # SVM
    # Define a focused parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # Initialize the SVM model
    model_svm = SVC(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model_svm, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')

    # Fit GridSearchCV on the training data
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Print the best hyperparameters
    print(f"Best hyperparameters: {grid_search.best_params_}")

    # Get the best model from grid search
    best_svm = grid_search.best_estimator_

    # Perform cross-validation with the best model
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_svm = cross_val_score(best_svm, X_train_resampled, y_train_resampled, cv=kf)

    print(f"SVM CV scores: {scores_svm}")
    print(f"Mean SVM CV score: {scores_svm.mean():.4f}")

    # Fit the best model on the training data
    best_svm.fit(X_train_resampled, y_train_resampled)

    # Predict on the test data
    y_pred = best_svm.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    # Return the trained model
    return best_svm
 
@app.route('/prediction_input')
def prediction_input():
    return render_template('prediction_input.html')

# @app.route('/result')
# def result():
#     # Load trained model
#     model = joblib.load('model.pkl')
#     # Make predictions
#     # Display results
#     return "Model trained and saved successfully!"


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        employee_id = request.form['employee_id']
        employee_type = request.form['EmployeeType']
        pay_zone = request.form['PayZone']
        department_type = request.form['DepartmentType']
        if (department_type == "Production"):
            department_type = "Production       "
        gender_code = request.form['GenderCode']
        race_desc = request.form['RaceDesc']
        marital_desc = request.form['MaritalDesc']
        # performance_score = request.form['Performance Score']
        current_employee_rating = int(request.form['current_employee_rating'])
        engagement_score = int(request.form['engagement_score'])
        satisfaction_score = int(request.form['satisfaction_score'])
        work_life_balance_score = int(request.form['work_life_balance_score'])

        input_data = {
            "EmployeeType": employee_type,
            "PayZone": pay_zone,
            "DepartmentType": department_type,
            "GenderCode": gender_code,
            "RaceDesc": race_desc,
            "MaritalDesc": marital_desc,
            # "Performance Score": performance_score,
            "current_employee_rating": current_employee_rating,
            "engagement_score": engagement_score,
            "satisfaction_score": satisfaction_score,
            "work_life_balance_score": work_life_balance_score
        }
        
         # Load the trained model
        model, label_encoders = joblib.load('model_and_encoders.pkl')

        # Preprocess categorical data
        input_data = preprocess_input(input_data , label_encoders)

        # Perform prediction
        print([list(input_data.values())])
        prediction = model.predict([list(input_data.values())])[0]
        if (prediction == 0):
            prediction = "Exceeds"
        elif (prediction == 1):
            prediction = "Fully Meets"
        elif (prediction == 2):
            prediction = "Needs Improvement"
        elif (prediction == 3):
            prediction = "PIP"
        
        # Render the prediction result template
        return render_template('prediction_result.html' , prediction = prediction)
    
# def preprocess_input(input_data):
#     # Initialize a LabelEncoder for each categorical feature
#     label_encoders = {
#         "pay_zone": LabelEncoder(),
#         "gender_code": LabelEncoder(),
#         "race_desc": LabelEncoder(),
#         "marital_desc": LabelEncoder(),
#         "employee_type": LabelEncoder(),
#         "department_type": LabelEncoder(),
#         "performance_score": LabelEncoder(),
#         # Add more LabelEncoders for other categorical features
#     }

#     # Transform categorical features into numerical representations
#     for feature, encoder in label_encoders.items():
#         print(input_data[feature])
#         input_data[feature] = encoder.fit_transform(input_data[feature])

#     return input_data
  
def preprocess_input(input_data, label_encoders):
    # Transform categorical features into numerical representations
    for feature, encoder in label_encoders.items():
        if feature in input_data:
            input_data[feature] = encoder.transform([input_data[feature]])[0]

    return input_data

if __name__ == '__main__':
    app.run(debug=True)

