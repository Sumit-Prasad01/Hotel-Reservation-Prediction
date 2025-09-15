import os
import pandas
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint
import mlflow
import mlflow.sklearn



logger = get_logger(__name__)

class ModelTrainig:

    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self. model_output_path =  model_output_path

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS


    def load_and_split_data(self):
        try:
            logger.info(f"loading data from {self.train_path}")
            train_df  = load_data(self.train_path)

            logger.info(f"loading data from {self.test_path}")
            test_df  = load_data(self.test_path)

            X_train = train_df.drop(columns = ['booking_status'])
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns = ['booking_status'])
            y_test = test_df['booking_status']

            logger.info("data splitted successfully for model training.")

            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error During loading data {e}")
            raise CustomException("Failed to load data",e)
        


    def tarin_lgbm(self,  X_train, y_train):
        try:
            logger.info("Initializing Our Model.")

            lgbm_model =  lgb.LGBMClassifier(random_state = self.random_search_params["random_state"])

            logger.info("Starting our HyperParameter Tuning")

            random_search = RandomizedSearchCV(
                    estimator = lgbm_model,
                    param_distributions = self.params_dist,
                    n_iter = self.random_search_params["n_iter"],
                    cv = self.random_search_params["cv"],
                    n_jobs = self.random_search_params["n_jobs"],
                    verbose = self.random_search_params["verbose"],
                    random_state = self.random_search_params["random_state"],
                    scoring = self.random_search_params["scoring"],
            )

            logger.info("Starting our HyperParameter Tunning")

            random_search.fit(X_train, y_train)

            logger.info("HyperParameter Tunning Completed")

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f"Best Parameters are {best_params}")

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error During training model {e}")
            raise CustomException("Failed to train model",e)
        

    def evaluate_mode(self, model, X_test, y_test):
        try:
            logger.info("Evaluating Our Model")

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1score = f1_score(y_test, y_pred)

            logger.info(f"Accuracy Score :{accuracy}")
            logger.info(f"Precision Score :{precision}")
            logger.info(f"Recall Score :{recall}")
            logger.info(f"F1 Score Score :{f1score}")


            return {
                "accuracy" : accuracy,
                "precision" : precision,
                "recall" : recall,
                "f1-score" : f1score
            }
        
        except Exception as e:
            logger.error(f"Error During Evaluating model {e}")
            raise CustomException("Failed to Evaluate model",e)
         

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            logger.info("Saving the model.")

            joblib.dump(model, self.model_output_path)
            logger.info(f"Model Saved to {self.model_output_path}")
        
        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to Save model",e)
        


    def run(self):
        try:
            with mlflow.start_run():

                logger.info("Starting Our MLflow Experimentation")

                logger.info("Starting our model training pipeline")

                logger.info("Logging that training and testing dataset to MLFlow")
                mlflow.log_artifact(self.train_path, artifact_path = 'datasets')
                mlflow.log_artifact(self.test_path, artifact_path = 'datasets')

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.tarin_lgbm(X_train, y_train)
                metrics = self.evaluate_mode(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)

                logger.info("Logging the model into mlflow")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging Params and metrics to Mlflow")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model Training Successfully Completed")
            
        except Exception as e:
            logger.error(f"Error while running model tarining pipeline {e}")
            raise CustomException("Failed to run tarin pipeline",e)
        

if __name__ == "__main__":
    trainer = ModelTrainig(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()