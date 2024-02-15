# this function opens the raw file and cleans up preliminary steps as needed

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# opens raw file / extracts usable columns 
def open_file():
    file_path = '../data/Debernardi et al 2020 data.csv'
    data = pd.read_csv(file_path)
    df = data[['age', 
             'sex',
             'plasma_CA19_9',
             'creatinine',
             'LYVE1',
             'REG1B',
             'TFF1',
             'REG1A',
             'diagnosis']].copy()
       
    # encoding and renaming 'sex' to 'is_male'
    encoder = OneHotEncoder(drop='first', sparse_output=False, dtype=int)  
    encoded_sex = encoder.fit_transform(df[['sex']]) 
    df['sex'] = encoded_sex
    df.rename(columns={'sex': 'is_male'}, inplace=True)
    
    # Return the DataFrame
    return df


# NOTE: 
# columns ('sample_id', 'patient_cohort', 'sample_origin', ' benign_sample_diagnosis') are not kept because they have no effect on the diagnosis of the patient
# column ('stage') may be used for future prediction tasks - however is left out for this model
# column ('diagnosis') will be used for prediction in the models