from modules.stt import *
from modules.FAQ_Script import *
from textblob import TextBlob
import time
import os

MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class AI():

    def __init__(self, stt_api="houndify", voice=0) -> None:
        # init FAQ module
        self.VirtualAssist = VirtualAssistant(0.79)
        self.VirtualAssist.train()

        # init STT module
        self.SpeechToText = SpeechToText()
        self.TranscriptAPI = stt_api
        self.VoiceOrText = voice # 0 for text only, 1 for voice & text

    def init_conversation(self, phone_number, priorknow_BOOL):
        """
            Initate call with provided phone number.

            Returns dict:
            [ call_duration, if_answered, prior_knowledge, [No. Questions User, No. Questions VA, [user_conversation]] ]
        """
        start_time = time.time()
        if_answered = 0
        prior_knowledge = 0
        qa_analysis = {"user_question_counter": 0, 
                       "va_question_counter": 0, 
                       "user_conversation": []}

        print("Calling " + str(phone_number))
        print("Initiating conversation...")
        print("")

        self.SpeechToText.print_dialogue_text("VA", "Hello. I am calling from University of Wollongong in Dubai.", self.VoiceOrText)
        self.SpeechToText.print_dialogue_text("VA", "Do you have some spare time to talk to me?", self.VoiceOrText)
        try:
            if (self.SpeechToText.yes_or_no(self.TranscriptAPI)):
                if_answered = 1

                # Prior Knowledge Block
                if not priorknow_BOOL:
                    self.SpeechToText.print_dialogue_text("VA", "Did you know about University of Wollongong In Dubai prior to this conversation?", self.VoiceOrText)
                    if (self.SpeechToText.yes_or_no(self.TranscriptAPI)):
                        self.SpeechToText.print_dialogue_text("VA", "Thats great to hear.", self.VoiceOrText)
                        prior_knowledge = 1
                    else:
                        self.SpeechToText.print_dialogue_text("VA", "That's alright. I can tell you a brief description about UOWD.", self.VoiceOrText)

                # Questions Block
                if priorknow_BOOL:
                    self.SpeechToText.print_dialogue_text("VA", "Do you have any more questions that'll help you decide if UOWD is the best choice for you?", self.VoiceOrText)
                else:
                    self.SpeechToText.print_dialogue_text("VA", "Do you have any questions that'll help you decide if UOWD is the best choice for you?", self.VoiceOrText)
                if (self.SpeechToText.yes_or_no(self.TranscriptAPI)):
                    self.SpeechToText.print_dialogue_text("VA", "Well, what do you have in mind? Say 'Bye' when you want to exit the call.", self.VoiceOrText)
                    self.answer_user_questions(qa_analysis)
                    print("")
                else:
                    self.SpeechToText.print_dialogue_text("VA", "Alright, no worries. Have a good day!", self.VoiceOrText)
                    print("")  
            else:
                self.SpeechToText.print_dialogue_text("VA", "No worries. Thank you for your time. Bye, have a good day!", self.VoiceOrText)
                print("")  
                
        except HangUp:
            self.SpeechToText.print_dialogue_text("VA", "Bye bye", self.VoiceOrText)
            print("")  
        except ConversationAttempt:
            self.SpeechToText.print_dialogue_text("VA", "No conversation attempt...Ending call", self.VoiceOrText) 
            self.SpeechToText.print_dialogue_text("VA", "Have a good day!", self.VoiceOrText)
            print("")   

        call_duration_sec = time.time() - start_time
        conversation_analysis = {"call_duration": call_duration_sec, 
                                 "answered": if_answered, 
                                 "prior_knowledge": prior_knowledge, 
                                 "qa_analysis": qa_analysis}
        return conversation_analysis 

    def answer_user_questions(self, qa_analysis):
        """
            Function to answer user questions. 

            Returns a list of call QA Analysis containing 
                User Questions Asked
                VA Questions Asked
                Successful User Conversation 
        """
        attempt_counter = 0
        user_question_counter = 0
        va_question_counter = 0
        user_conversation = []

        while True:
            # Exit call if no communication attempts twice in a row
            if attempt_counter == 3:
                self.SpeechToText.print_dialogue_text("VA", "No conversation attempt twice...", self.VoiceOrText)
                self.SpeechToText.print_dialogue_text("VA", "Ending call....", self.VoiceOrText)
                break
            try:
                user_query = self.SpeechToText.transcribe_audio(self.TranscriptAPI)
            except sr.WaitTimeoutError:  # Conversation attempt
                attempt_counter += 1
                if attempt_counter < 3:
                    self.SpeechToText.print_dialogue_text("VA", "Are you still there?", self.VoiceOrText)
                continue
            except HangUp: # Loop break condition
                self.SpeechToText.print_dialogue_text("VA", "Bye Bye!", self.VoiceOrText)
                break
            except TranscriptError:
                self.SpeechToText.print_dialogue_text("VA", "Couldn't quite get that. Could you repeat?", self.VoiceOrText)
                continue
            
            Query_result = self.VirtualAssist.pred_ans(user_query)
            self.SpeechToText.print_dialogue_text("USER", user_query) 
            self.SpeechToText.print_dialogue_text("VA",  Query_result["result"], self.VoiceOrText)

            # Successful answer
            if Query_result["success"] == 1:
                user_conversation.append(user_query)

            # Check if VA has a question opportunity
            VA_question = self.va_question_opportunity(user_query)
            if VA_question["success"] == 1:
                self.SpeechToText.print_dialogue_text("VA", VA_question["question"], self.VoiceOrText)
                attempt = 0
                while True:
                    if attempt == 3:
                        self.SpeechToText.print_dialogue_text("VA", "No conversation attempt twice...", self.VoiceOrText)
                        user_conversation.append("")
                        break
                    try:
                        User_Answer = self.SpeechToText.transcribe_audio(self.TranscriptAPI)
                        user_conversation.append(User_Answer)
                        break
                    except sr.WaitTimeoutError:  # Conversation attempt
                        attempt += 1
                        if attempt < 3:
                            self.SpeechToText.print_dialogue_text("VA", "Are you still there?", self.VoiceOrText)
                        continue
                    except TranscriptError:
                        self.SpeechToText.print_dialogue_text("VA", "Couldn't quite get that. Could you repeat?", self.VoiceOrText)
                        continue
                self.SpeechToText.print_dialogue_text("VA", "Alright. Any more questions?", self.VoiceOrText)

            user_question_counter += Query_result["success"]  
            va_question_counter += VA_question["success"]         
            attempt_counter = 0
        
        qa_analysis["user_question_counter"] = user_question_counter, 
        qa_analysis["va_question_counter"] = va_question_counter,
        qa_analysis["user_conversation"] = user_conversation

    def va_question_opportunity(self, user_query):
        """
            Finds if the current User Query allows the Virtual Assistant to follow up with a question.
            Gets the user answer on that question for post-conversation sentimental analysis 
        """
        question_set = pd.read_csv("files/follow_up_Q.csv")
        question_set['Q'] = question_set['Q'].pipe(remove_punctuation).pipe(remove_digits)

        question_set_embeds = MODEL.encode(question_set['Q'])
        user_query_embeds = MODEL.encode(user_query)

        similarity_match = util.pytorch_cos_sim(user_query_embeds, question_set_embeds)
        closest_index = np.argmax(similarity_match)

        if similarity_match[0][closest_index].item() > 0.79:
            return {"question": question_set['F_Q'][closest_index.item()], "success": 1}
        
        return {"question": "", "success": 0}

    def get_sentiment_polarity(self, text):
        return TextBlob(text).sentiment.polarity

    def record_conversation(self, user_conversation, phone_number):
        with open("conversations/" + str(phone_number) + ".txt", "a+") as f:
            for line in user_conversation:
                f.write(line)
                f.write("\n")

    def sentimental_analysis(self, phone_number):
        file = open("conversations/" + str(phone_number) + ".txt", "r")
        lines = file.readlines()

        total = 0
        for i in lines:
            total += self.get_sentiment_polarity(i)

        if len(lines) == 0:
            return 0
        return total / len(lines)


    def run_app(self) -> None:
        """
            Loops through Student Details CSV and initates calls.
            Records conversation details on:
                Sentimental Score
                Length of Talk in Minutes
                Number of Communications
                Priror Awarness
                No of questions User asked
                No of questions Virtual Assistant asked
            for analysis on lead scoring
        """
        student_df = pd.read_csv("files/small_set.csv")
        priorKnowledge = False

        for i in student_df.index:
            number = str(student_df.at[i, 'MOBILE'])
            if student_df.at[i, 'No. comunications'] > 0:
                priorKnowledge = True

            analysis = self.init_conversation(number, priorKnowledge)
            self.record_conversation(analysis["qa_analysis"]["user_conversation"], number)

            student_df.at[i, 'Tone or sentimental score'] = self.sentimental_analysis(number)
            student_df.at[i, 'Length of talk (Mins)'] += analysis["call_duration"]/60
            student_df.at[i, 'No. comunications'] += analysis["answered"]
            student_df.at[i, 'Prior awareness'] = analysis["prior_knowledge"]
            student_df.at[i, 'No. questions student'] += analysis["qa_analysis"]["user_question_counter"]
            student_df.at[i, 'No. questions VA'] += analysis["qa_analysis"]["va_question_counter"]

        # Write updated CSV to file
        student_df.to_csv("files/small_set.csv", index=False)
        return None

    def reset_leads(self):
        """
            Debug purposes to quickly reset leads
            and delete conversations
        """
        student_df = pd.read_csv("files/small_set.csv")
        for i in student_df.index:
            student_df.at[i, 'Tone or sentimental score'] = 0
            student_df.at[i, 'Length of talk (Mins)'] = 0
            student_df.at[i, 'No. comunications'] = 0
            student_df.at[i, 'Prior awareness'] = 0
            student_df.at[i, 'No. questions student'] = 0
            student_df.at[i, 'No. questions VA'] = 0

            number = str(student_df.at[i, 'MOBILE'])

            try:
                os.remove("conversations/" + number + ".txt")
            except FileNotFoundError:
                pass

            student_df.to_csv("files/small_set.csv", index=False)
