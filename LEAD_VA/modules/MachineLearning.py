import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ibm_watson_machine_learning import APIClient
from KEYS import IBM_ML_API_KEY, IBM_ML_LOCATION

# UNFINISHED
class LeadProcessing():

    def process_leads(self, csv_location):
        leads_df = pd.read_csv(csv_location)

        X = leads_df[['Length of talk (Mins)', 'Tone or sentimental score',
       'No. comunications', ' Prior awareness', 'No. questions student',
       'No. questions VA']]
        Y = leads_df['Lead Score']

        X_train, Y_train = train_test_split(X, Y, test_size=0.3, random_state=101)
        logm = LogisticRegression()
        logm.fit(X_train, Y_train)
        api_key = IBM_ML_API_KEY
        location = IBM_ML_LOCATION

        wml_credentials = {
            "apikey": api_key,
            "url": 'https://'+ location + '.ml.cloud.ibm.com'
        }
        client = APIClient(wml_credentials)
        client.set.default_space("d37da41d-2432-4563-9df6-f8c820382373")

        sofware_spec_uid = client.software_specifications.get_id_by_name("default_py3.9")
        metadata = {
            client.repository.ModelMetaNames.NAME: 'Lead Scoring Test',
            client.repository.ModelMetaNames.TYPE: 'scikit-learn_0.23',
            client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sofware_spec_uid
        }

        published_model = client.repository.store_model(
            model=logm,
            meta_props=metadata,
            training_data=leads_df[['Length of talk (Mins)', 'Tone or sentimental score',
            'No. comunications', ' Prior awareness', 'No. questions student',
            'No. questions VA']],
            training_target=leads_df['Lead Score']
        )

        # return CSV