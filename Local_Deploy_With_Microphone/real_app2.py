import streamlit as st
import datetime
import time
import regex as re
#import SessionState

### Config
st.set_page_config(
    page_title="Web-detection",
    layout="wide"
)

### HEADER
st.title('Web Application to detect violents speeches')
st.header(" A  machine learning application to detect violent or non-violent speech")
st.markdown(""" This application allow to classify violent ou non-violent message.
 It attributes a score to every sentence. The purpose of this web application is to fight against 
 the verbal violences at school.  
""")
st.title("How to use it")
st.markdown("* First chose an interval of time.")
st.markdown("* After, click on the button 'Start recording'.")
st.markdown ("* After that, the app will register all the conversations that it can \
listen.")
st.markdown("* It will give a hate score to every sentence and an average value\
    of hate score during the interval of time.")

st.markdown("* If you want to see a demo, click 'Demo'.")    


st.subheader('Detection violent speeches')


import speech_recognition as sr

def user_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source,phrase_time_limit= 30)
        try:
            text = r.recognize_google(audio)
            text = re.sub(r'f[*]+','fuck',text)
            text = re.sub(r's[*]+','suck',text)
            return text
        except Exception as e:
            return e


st.sidebar.header("Time of the speech")
lt = time.localtime()
start_datetime = datetime.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min)
start_date = st.sidebar.date_input('start date', start_datetime)
start_time = st.sidebar.text_input("start time", start_datetime.strftime("%H:%M"))

end_datetime = start_datetime
end_date = st.sidebar.date_input('end date', end_datetime)
end_time = st.sidebar.text_input("end time", end_datetime.strftime("%H:%M"))
start_date = start_date.strftime("%Y:%m:%d")
start_datetime = (start_date + ":" + start_time).split(":")
tuple_start_datetime = tuple([int(time) for time in start_datetime])
start_datetime = datetime.datetime(*tuple_start_datetime)
end_date = end_date.strftime("%Y:%m:%d")
end_datetime = (end_date + ":" + end_time).split(":")
tuple_end_datetime = tuple([int(time) for time in end_datetime])
end_datetime = datetime.datetime(*tuple_end_datetime)
lt = time.localtime()

current_datetime = datetime.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min)

if st.button("Demo"):
   import pandas as pd
   df_demo = pd.read_excel("demo_succession.xlsx")

   score_hate_mean = int(100*df_demo.iloc[-1,-3:].mean()+0.5)/100

   st.write("From https://www.youtube.com/watch?v=Ut4pgRu5gHU")

   st.write(f"The hate score is {score_hate_mean}")

   from io import BytesIO


   def to_excel(df):
      output = BytesIO()
      writer = pd.ExcelWriter(output, engine='xlsxwriter')
      df.to_excel(writer, index=False, sheet_name='Sheet1')
      workbook = writer.book
      worksheet = writer.sheets['Sheet1']
      format1 = workbook.add_format({'num_format': '0.00'}) 
      worksheet.set_column('A:A', None, format1)  
      writer.save()
      processed_data = output.getvalue()
      return processed_data
   df_xlsx2 = to_excel(df_demo)
   st.download_button(label='📥 Download Demo Results',
                                data=df_xlsx2 ,
                                file_name= 'demo_results.xlsx')   

   st.write(df_demo)



if st.button("Start recording"):  
    sentences = []
    date_sentences = []
    st.write(f"It bigins at {start_datetime} and it ends at {end_datetime}")
    nb_sentences = 0
    while current_datetime <= end_datetime:
        lt = time.localtime()
        current_datetime = datetime.datetime(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min)

        if current_datetime >= start_datetime:
            sentence = user_input()
            if type(sentence) == str:
                sentences.append(sentence)
                date_sentences.append(current_datetime)
                nb_sentences += 1
                st.write(f"Sentence {nb_sentences}")
            else:
                st.write("No sound detected")
    import pandas as pd            
    df = pd.DataFrame(columns=['Time','Sentences'])
    df['Sentences'] = sentences
    df['Time'] = date_sentences
    import pickle
    import numpy as np
    import en_core_web_md
    nlp = en_core_web_md.load()
    from keras.models import load_model
    with open('TFIDF.pkl','rb') as TFIDF:
        tfidf = pickle.load(TFIDF)
    
    with open('LR.pkl','rb') as LR:
     lr = pickle.load(LR)

    def pred_prob_LR(text):
     token_text = nlp(text)
     text = [element.lemma_.lower() for element in token_text]
     text = " ".join(text)
     text_tfidf = tfidf.transform([text])
     probs = lr.predict_proba(text_tfidf)
     return probs[0][1]

    with open('xgboost.pkl','rb') as XGBOOST:
     xgboost = pickle.load(XGBOOST) 

    def pred_prob_XGBOOST(text):
    
     token_text = nlp(text)
     text = [element.lemma_.lower() for element in token_text]
     text = " ".join(text)
     text_tfidf = tfidf.transform([text])
     probs = xgboost.predict_proba(text_tfidf)
     return probs[0][1]

    with open('tokenizer.pkl','rb') as TOKEN:
     tokenizer = pickle.load(TOKEN)
    
    model = load_model("neuron_network.h5")    

    nb_features = int(pd.read_csv('X_train_shape_1.txt').columns.tolist()[0])

    def pred_prob_NN(text):
     token_text = nlp(text)
     text = [element.lemma_.lower() for element in token_text]     
     token_text = tokenizer.texts_to_sequences([text])
     enter = np.zeros((1,nb_features))
     enter[0,:len(token_text[0])] = token_text[0]
     probs = model.predict(enter)
     return probs[0][0]      


    df['Hate score by LR'] = df['Sentences'].apply(pred_prob_LR)
    df['Hate score by LR'] = df['Hate score by LR'].apply(lambda x : round(x,2))
    df['Hate score by XGBOOST'] = df['Sentences'].apply(pred_prob_XGBOOST)
    df['Hate score by XGBOOST'] = df['Hate score by XGBOOST'].apply(lambda x : round(x,2))
    df['Hate score by Network Neural'] = df['Sentences'].apply(pred_prob_NN)
    df['Hate score by Network Neural'] = df['Hate score by Network Neural'].apply(lambda x : round(x,2))
    try:
     dict_means = {col:int(100*df[col].mean()+0.5)/100 for col in df.columns.tolist()[-3:]}
    except ValueError:
     dict_means = {col:np.nan for col in df.columns.tolist()[-3:]}    
    dict_means['Sentences'] = "Average value"
    dict_means = [dict_means]
    df = pd.concat([df,pd.DataFrame(dict_means)],axis = 0)
    try:
     score_hate_mean = int(100*df.iloc[-1,-3:].mean()+0.5)/100
    except:
     score_hate_mean = np.nan    
    st.write(f'The hate score during this time is {score_hate_mean}')

    from io import BytesIO
    from pyxlsb import open_workbook as open_xlsb
    

    def to_excel(df):
      output = BytesIO()
      writer = pd.ExcelWriter(output, engine='xlsxwriter')
      df.to_excel(writer, index=False, sheet_name='Sheet1')
      workbook = writer.book
      worksheet = writer.sheets['Sheet1']
      format1 = workbook.add_format({'num_format': '0.00'}) 
      worksheet.set_column('A:A', None, format1)  
      writer.save()
      processed_data = output.getvalue()
      return processed_data
    df_xlsx = to_excel(df)
    st.download_button(label='📥 Download Results',
                                data=df_xlsx ,
                                file_name= 'results.xlsx')

    st.write(df) 



