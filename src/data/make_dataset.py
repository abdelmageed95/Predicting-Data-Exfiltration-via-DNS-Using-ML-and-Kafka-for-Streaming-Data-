# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from src.models import predict_model
from kafka import KafkaConsumer
from dotenv import find_dotenv, load_dotenv
import warnings
warnings.filterwarnings("ignore")


output_filepath = r"D:\assignment2\assignment2-abdelmageed95\data\processed"
limit = 100000
def main(output_filepath ,limit):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # for training model uncomment the next 5 line
    # training_data_path = r"D:\assignment2\assignment2-abdelmageed95\src\data\training_dataset.csv"
    # df = prepare_df(training_data_path)
    # train_model = training(df)
    # if train_model == "done":
    #     print("Done")


    # read data from kafka topic 
    consumer = KafkaConsumer('ml-raw-dns', bootstrap_servers='localhost:9092', auto_offset_reset='earliest')
    output = []  #list to append the prediction dataframes 
    for index ,message in enumerate(consumer):
        if  index < limit : # number of messages to read from kafka topics
            url = message.value.decode("utf-8") # read the url from the message 
            df = predict_model.generate_df(url) # function that read url , extract features , predict the label with confidence score ,and return them in a row df
            output.append(df) # append the row df to the output list
            print( index , df)
        else :
            break

    predict_model.save_data(output ,output_filepath) # function that concatenate the dfs into big data frame and save the output to csv file 


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(output_filepath, limit)
