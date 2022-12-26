import speech_recognition as sr
import pyttsx3 as tts
from ibm_watson import SpeechToTextV1, TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
import numpy as np
from KEYS import *

MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')
YES = ['yep', 'yea', 'yes', 'yeap', 'yeah', 'affirmative', 'fine', 'okay', 'true', 'alright', 'all right', 'aye', 'by all means', 'certainly', 'definitely', 'exactly', 'gladly', 'most assuredly', 'naturally', 'of course', 'ofcourse', 'positively', 'sure thing', 'surely', 'undoubtedly', 'very well', 'without fail']
YES_EMBEDS = MODEL.encode(YES)




class ConversationAttempt(Exception):
    def __init__(self):
        self.message = "USER_INPUT_ERROR: No attempts at communication made"
    def __str__(self) -> str:
        return self.message

class TranscriptError(Exception):
    def __init__(self) -> None:
        self.message = "TRANSCRIPTION_ERROR: Could not transcribe"
    def __str__ (self):
        return self.message

class HangUp(Exception):
    def __init__(self) -> None:
        self.message = "EXIT PHRASE DETECTED"
    def __str__(self):
        return self.message

class SpeechToText():

    def print_dialogue_text(self, agent, dialgoue, voice=0):
        print(agent + ": " + dialgoue)
        if voice == 1:
            self.speak(dialgoue)

    def transcribe_audio(self, api) -> str:
        """
            Transcribe user input using specified api
            Default API: Houndify

            Raises: ``TranscriptError`` if invalid transcriptor or transcript error found, ``HangUp`` if Bye is heard
        """
        if api.lower() == "ibm":
            authenticator = IAMAuthenticator(IBM_STT_API_KEY)
            ibm_stt = SpeechToTextV1(authenticator=authenticator)
            ibm_stt.set_service_url(IBM_STT_URL)

        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)

            # Get audio block
            self.print_dialogue_text("SYSTEM", "Listening....")
            user_audio = r.listen(source, timeout=10)

            # Transcribe Audio Block
            self.print_dialogue_text("SYSTEM", "Reocognizing using: " + api)

            if api.lower() == "houndify":
                user_query = r.recognize_houndify(user_audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY)
            elif api.lower() == "ibm":
                data_flac = user_audio.get_flac_data()
                result = ibm_stt.recognize(data_flac, content_type="audio/flac", model="en-WW_Medical_Telephony").get_result()
                try:
                    user_query = result['results'][0]['alternatives'][0]['transcript']
                except IndexError:
                    raise TranscriptError
            else:
                raise TranscriptError

            if "bye" in user_query:
                raise HangUp

            return user_query

    def yes_or_no(self, stt_api) -> bool:
        """
            Getting user choice on a 
            yes or no question
        """
        attempts = 0
        yes_embs = MODEL.encode("yes")

        while True:
            # End conversation if
            # no communication attempt made
            if attempts == 3:
                raise ConversationAttempt

            try:
                user_answer = self.transcribe_audio(stt_api)
            except sr.WaitTimeoutError:
                attempts += 1
                if attempts < 3:
                    self.print_dialogue_text("VA", "Are you there?")
                continue

            # Filter out all words that aren't synonyms of "yes"
            # and generate embeds
            user_answer_tokens = word_tokenize(user_answer)
            user_answer_filtered = ' '.join([str(item) for item in user_answer_tokens if item in YES])
            user_answer_filtered_embeds = MODEL.encode(user_answer_filtered)

            similarity_array = util.pytorch_cos_sim(user_answer_filtered_embeds, YES_EMBEDS)
            closest_match_index = np.argmax(similarity_array)
            
            if (similarity_array[0][closest_match_index].item() > 0.9):
                return True
            return False

    def speak(self, string):
        engine = tts.init('sapi5')
        engine.setProperty('voice', 'voices[0].id')

        engine.say(string)
        engine.runAndWait()
