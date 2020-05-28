import sys, getopt
import os

import numpy as np
import pandas as pd

from patient_data import Patient

PATH_TO_CSV_TABLE = './dataset.csv'

def arguments_parsing(argv):
    path_to_raw_data = ''
    path_to_preprocessed_data = ''
    try:
        opts, args = getopt.getopt(argv, "h", ["path_to_raw_data=", "path_to_preprocessed_data="])
    except getopt.GetoptError:
        print("./data_preprocessing.py --path_to_raw_data=<str> --path_to_preprocessed_data=<str>")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print("./data_preprocessing.py --path_to_raw_data=<str> --path_to_preprocessed_data=<str>")
            sys.exit()
        elif opt == "--path_to_raw_data":
            path_to_raw_data = arg
        elif opt == "--path_to_preprocessed_data":
            path_to_preprocessed_data = arg
            
    return path_to_raw_data, path_to_preprocessed_data

def make_csv_table(dataset, path_to_csv_file):
    pd.DataFrame(dataset, columns=['image', 'mask', 'frame']).to_csv(path_to_csv_file, index=False)

def main(argv):
    path_to_raw_data, path_to_preprocessed_data = arguments_parsing(argv)
    paths_to_patients = [os.path.join(path_to_raw_data, patient_name) for patient_name in os.listdir(path_to_raw_data)]

    dataset = np.empty((0, 3))
    for path_to_patient in paths_to_patients:
        patient = Patient(path_to_patient)
        patient_name = patient.get_patient_name()
    
        print('{} data reading ...'.format(patient_name)) 
        patient_data = patient.get_patient_data()
    
        print('{} data preprocessing ...'.format(patient_name))
        patient.data_preprocessing(patient_data)
    
        print('{} preprocessed data saving ...'.format(patient_name))
        patient.save_tiff_images(patient_data, path_to_preprocessed_data)
    
        print()
        
        dataset = np.vstack((dataset, patient.make_dataset_table(patient_data, path_to_preprocessed_data)))
    
    print('dataset csv table creating...')
    make_csv_table(dataset, PATH_TO_CSV_TABLE)

if __name__ == "__main__":
    main(sys.argv[1:])
