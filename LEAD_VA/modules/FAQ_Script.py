# Reading csv
import numpy as np
import pandas as pd

# Cleaning text
from lingualytics.preprocessing import remove_lessthan, remove_punctuation  #, remove_stopwords
from texthero.preprocessing import remove_digits

# Sentence transformer models to generate embeds
# util for similarity engine
from sentence_transformers import SentenceTransformer, util

# Model manipulation
import pickle
import glob

# Disable warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class VirtualAssistant():
    
    def __init__(self, sim_match: int) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2") # load sentence transformer model
        self.df = pd.DataFrame() # init empty DF
        self.similarity_match_requirement = sim_match 
    
    def train(self, csv_holder="training_csv"):
        """
            Specify a parent folder containing Q&A files as CSV
            Automatically imports them into the dataframe
            and writes the embeddings for the question set
            to a .pickle file
        """
        # Get all Q&A CSV in folder and generate dataframe
        CSV_list = glob.glob(csv_holder+"/*.csv")
        self.df = pd.concat((pd.read_csv(f) for f in CSV_list), ignore_index=True)
        
        # Clean question set
        self.df['clean_Q'] = self.df['Q'].pipe(remove_digits).pipe(remove_punctuation)
        
        # Generate embeds and pickle in binary
        q_embs = self.model.encode(self.df['clean_Q'])
        pickle.dump(q_embs, open("models\\model_va.pickle", "wb"))

    def pred_ans(self, usr_query: str):
        """
            Predicts answer based on user query
            Question set converted to embeds in .train() which is loaded here

            Convert user query to embeds and compare to question set 
            to find best match, returning answer
        """
        # open the pickle jar and load initial Q&A embeds
        q_embs = pickle.load(open("models\\model_va.pickle", "rb")) 

        # Convert usr query to df and clean using same pipline
        # generate sentence embeddings for usr query
        usr_qu_df = pd.DataFrame([usr_query], columns=['query'])
        usr_qu_df['clean_query'] = usr_qu_df['query'].pipe(remove_digits).pipe(remove_punctuation)
        usr_qu_embeds = self.model.encode(usr_qu_df['clean_query'])

        # find closest match to question dataframe and get index
        simi_tensor = util.pytorch_cos_sim(usr_qu_embeds, q_embs)
        closest_index = np.argmax(simi_tensor) 
        
        # if question match not close enough
        # ask to repeat
        # return a list of ans + if question had a successful answer
        if simi_tensor[0][closest_index] >= self.similarity_match_requirement:
            return {"result": self.df['A'][closest_index.item()], "success": 1}
        else:
            return {"result": "Im not sure I quite understand. Can you repeat your question?", "success": 0}

    def run_app(self):
        """
            Standalone function to init and use
            this component on its on
        """
        while True:
            usr_query = input("Ask a query(or type 'exit' to exit): ")
            if usr_query.lower() == "exit":
                break
            else:
                response = self.pred_ans(usr_query)
                print(response)
                print("-----------------")

# change user input to match to previous question
# and check if related