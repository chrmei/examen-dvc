# Evaluation

This evaluation consists of four parts. We will work with a dataset containing information about mineral processing, specifically focusing on the flotation process used to concentrate silica from ore. The dataset includes various operational parameters and their impact on silica concentration, which is the target variable. The dataset you will download contains the following information:

- **ave_flot_air_flow**: Average air flow rate in the flotation process.
- **ave_flot_level**: Average level in the flotation cells.
- **iron_feed**: Amount of iron ore entering the flotation process.
- **starch_flow**: Flow rate of starch used as a reagent in the flotation process.
- **amina_flow**: Flow rate of amine used as a collector in the flotation process.
- **ore_pulp_flow**: Flow rate of ore pulp.
- **ore_pulp_pH**: pH level of the ore pulp, which can affect the flotation process.
- **ore_pulp_density**: Density of the ore pulp, another critical parameter in the flotation process.
- **silica_concentrate**: Silica concentration in the final product, which is the target variable.

The objective of the modeling is to understand and model how different operational parameters affect the silica concentration in the flotation process. The dataset contains 1817 entries, all columns are of type float64, and there are no duplicates or missing values.

The data is accessible through the following link: https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.

Regarding the goal of this exam, it is to set up a modeling workflow using DVC and DagsHub. The final submission will be in the form of a repository on DagsHub, which you will share with https://dagshub.com/licence.pedago, adding them as a collaborator with read-only access rights. On the platform, you will submit a .zip file containing a .md file with your name, first name, email address, and the link to your DagsHub repository.

We start by forking and cloning the repository: https://github.com/DataScientest-Studio/examen-dvc. You will notice it contains the architecture that the project should have. The directory structure resembles:

```
├── examen_dvc          
│   ├── data       
│   │   ├── processed      
│   │   └── raw       
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src       
│   └── README.md.py       
```

Create a virtual environment where you will work throughout the exam.

## 1. Script Creation

The first step is to build the necessary scripts for the workflow of this modeling project. We expect to have at least 5 scripts, each targeting the following steps:

- **Data Splitting**: Split the data into training and testing sets. Our target variable is silica_concentrate, located in the last column of the dataset. This script will produce 4 datasets (X_test, X_train, y_test, y_train) that you can store in data/processed.

- **Data Normalization**: As you may notice, the data varies widely in scale, so normalization is necessary. You can use existing functions to construct this script. As output, this script will create two new datasets (X_train_scaled, X_test_scaled) which you will also save in data/processed.

- **GridSearch for Best Parameters**: Decide on the regression model to implement and the parameters to test. At the end of this script, we will have the best parameters saved as a .pkl file in the models directory.

- **Model Training**: Using the parameters found through GridSearch, we will train the model and save the trained model in the models directory.

- **Model Evaluation**: Finally, using the trained model, we will evaluate its performance and make predictions. At the end of this script, we will have a new dataset in data containing the predictions, along with a scores.json file in the metrics directory that will capture evaluation metrics of our model (e.g., MSE, R2).

Each script should be located in the corresponding folder (src/data or src/models). If necessary, you may add additional scripts. Ensure that your scripts have clear and relevant names.

## 2. Connect Your Repository to DagsHub

If not already done and if you are working on GitHub, connect this repository to your DagsHub account. Then, set DagsHub as your remote location for data tracking. Don't forget to adjust your .gitignore based on your needs.

## 3. DVC Pipeline

Using the DVC commands covered in the course, set up a pipeline that reproduces the workflow of your model. Make sure to utilize the scripts you created in step 1.

## 4. Submission

To submit the exam on the platform, you will send a .zip file containing an .md file with your name, first name, email address, and the link to your DagsHub repository. Then, share your repository with https://dagshub.com/licence.pedago, adding them as a collaborator with read-only access. To validate the exam, we expect to find in this repository:

- The 5 preprocessing, modeling, and model evaluation scripts detailed in step 1.
- A .dvc folder with a config file specifying information about the remote location.
- A .pkl file in the _models_ tab of DagsHub with the trained model.
- A .json file in the metrics folder with the evaluation metrics of the model.
- A dvc.yaml file with the DVC pipeline steps, along with a dvc.lock file containing the backup information.
- The _data_ tab should display the data properly.
- The DagsHub pipeline schema should resemble:

![DagsHub Pipeline Schema](Schema_exam.png)
