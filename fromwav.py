import streamlit as st
from pydub import AudioSegment
from pydub.silence import split_on_silence
import regex as re
import spacy
from keras.models import load_model
import xgboost

st.write(xgboost.__version__)

nlp = spacy.load("en_core_web_sm")

import speech_recognition as sr
r = sr.Recognizer()

wav_file = st.file_uploader("Upload wav",type = ["wav"])

sentences = []   

try:
    sound_file = AudioSegment.from_wav(wav_file)
    audio_chunks = split_on_silence(sound_file, min_silence_len=500, silence_thresh=-40 )
    list_chuncks = []
    for i, chunk in enumerate(audio_chunks):
     out_file = "chunk{0}.wav".format(i)
     chunk.export(out_file, format="wav")
     list_chuncks.append(out_file)
    for wave_sentence_name in list_chuncks: 
     with sr.AudioFile(wave_sentence_name) as source:
      audio = r.listen(source)
     try:
      text = r.recognize_google(audio)
      text = re.sub(r'f[*]+','fuck',text)
      text = re.sub(r's[*]+','suck',text)      
      sentences.append(text)
     except Exception as e:
      st.write("Exception: en_core_web_md"+str(e))
    import pandas as pd            
    df = pd.DataFrame(columns=['Sentences'])
    df['Sentences'] = sentences
    import pickle
    import numpy as np
#    import en_core_web_md
#    nlp = en_core_web_md.load()
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


except Exception as e:
    st.write(e) 

