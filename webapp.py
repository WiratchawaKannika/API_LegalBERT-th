import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore') 

import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pythainlp 
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize # ใช้ในการตัดคำ
from pythainlp.corpus import wordnet
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from stop_words import get_stop_words
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer
from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify, request
import numpy as np
from numpy import load
import json
import pandas as pd


app = Flask(__name__)
run_with_ngrok(app)


nltk.download('words')
th_stop = ' '.join(list(thai_stopwords()))
#th_stop = tuple(thai_stopwords('thai'))
en_stop = tuple(get_stop_words('en'))
p_stemmer = PorterStemmer()

pretrained = "monsoon-nlp/bert-base-thai"#@param ["monsoon-nlp/bert-base-thai"] 
model_path = 'C:\\Users\\LENOVO\\Desktop\\Webapp_LegalBERTth\\model_finetuning\\save_model\\1611409445'
session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=session_config)
tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], model_path)
name = pretrained
#tokenizer = AutoTokenizer.from_pretrained(name)
#model = AutoModel.from_pretrained(name)
tz = BertTokenizer.from_pretrained(name)

def split_word(text):
    tokens = word_tokenize(text, engine="newmm")
    tokens = [i for i in tokens if not i in th_stop and not i in en_stop] # Remove stop words ภาษาไทย และภาษาอังกฤษ 
    tokens = [p_stemmer.stem(i) for i in tokens] # หารากศัพท์ภาษาไทย และภาษาอังกฤษ # English
    tokens_temp=[]  # Thai
    for i in tokens:
        w_syn = wordnet.synsets(i)
        if (len(w_syn)>0) and (len(w_syn[0].lemma_names('tha'))>0):
            tokens_temp.append(w_syn[0].lemma_names('tha')[0])
        else:
            tokens_temp.append(i)
    tokens = tokens_temp
    tokens = [i for i in tokens if not i.isnumeric()] # ลบตัวเลข
    tokens = [i for i in tokens if not ' ' in i] # ลบช่องว่าง
    return tokens


def predict_text(input_value):
    e=split_word(input_value) #ตัดคำ
    #Extrac Feature input for Prediction
    #tz = BertTokenizer.from_pretrained(name)
    # The senetence to be encoded
    # Encode the sentence
    encoded = tz.encode_plus(
        text=e,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length = 128,  # maximum length of a sentence
        pad_to_max_length=True,  # Add [PAD]s
        return_attention_mask = True,  # Generate the attention mask
        return_tensors = 'pt',  # ask the function to return PyTorch tensors
)
        # Get the input IDs and attention mask in tensor format
    input_ids = encoded['input_ids']
    input_ids_new = np.array((input_ids),dtype = 'int64')
    softmax_tensor = sess.graph.get_tensor_by_name('loss/Softmax:0')
    predictions = sess.run(softmax_tensor, {'Placeholder:0': input_ids_new})
    labels = np.array([['Personal Rights', 'family', 'labor', 'contract', 'criminal']])
    for i in range(predictions.shape[1]):
        score = predictions.max()
        if predictions[0,i] == score :
            tag = labels[0,i]
            tag = str(tag)
    print(predictions)
    print((score*100))
    return tag, input_ids_new


def EuclideanDistance(input_ids_new, all_fea):
    Distance = np.sqrt((np.square(input_ids_new[:,np.newaxis]-all_fea).sum(axis=2)))
    #numpy.ndarray to list
    list_Distance = Distance.tolist()
    list_Distance_2 =[x for xs in list_Distance for x in xs] 
    return list_Distance_2 #return list of number edu distance


def create_table(list_question, list_anws, list_dist):
    #tag_data = Data.loc[Data['tag'] == 'กฎหมายอาญา(Criminal)'].reset_index(drop=True)
    #message_list = tag_table['message'].to_list()
    tabel_feature = pd.DataFrame(list(zip(list_question,list_anws, list_dist)), 
                      columns =['question', 'answer', 'distance'])
    return tabel_feature


def create_df_similarity(tabel_feature):
    tabel_feature_sort = tabel_feature.sort_values(by=['distance']).reset_index(drop=True)
    similarity_rank3 = tabel_feature_sort[0:3]
    question_similarity_rank3 = similarity_rank3['question']
    anws_similarity_rank3 = similarity_rank3['answer']
    question_similar1 = question_similarity_rank3[0] 
    anws_similar1 = anws_similarity_rank3[0] 
    question_similar2 = question_similarity_rank3[1] 
    anws_similar2 = anws_similarity_rank3[1]
    question_similar3 = question_similarity_rank3[2]
    anws_similar3 = anws_similarity_rank3[2]
    return question_similar1, anws_similar1, question_similar2, anws_similar2, question_similar3, anws_similar3


def text_similarity(tag_law, input_ids_new):
    #'%s' % tag_law 
    #read feature file
    feature_all_tag = np.load("%s.npy" % tag_law) 
    #calculate Distance
    list_dist = EuclideanDistance(input_ids_new, feature_all_tag)
    #create dataframe
    df = pd.read_csv('%s.csv' % tag_law)    
    list_question = df['คำถาม'].to_list()
    list_anws = df['คำตอบ'].to_list()
    tabel_feature = create_table(list_question, list_anws, list_dist)
    #sort Euclidean distance
    question_similar1, anws_similar1, question_similar2, anws_similar2, question_similar3, anws_similar3 = create_df_similarity(tabel_feature)  
    return question_similar1, anws_similar1, question_similar2, anws_similar2, question_similar3, anws_similar3  


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_value = request.form["text"]
        tag_law, input_ids_new = predict_text(input_value)
        question_similar1, anws_similar1, question_similar2, anws_similar2, question_similar3, anws_similar3 = text_similarity(tag_law, input_ids_new)
        #print(input_value)
        print(tag_law)
        #print(question_similar1)
        #print(anws_similar1)
        #print(question_similar2)
        #print(anws_similar2)
        #print(question_similar3)
        #print(anws_similar3)
        json_data = json.dumps({'input_text':input_value, 'result_tag':tag_law, 'result_Q1':question_similar1, 'result_A1':anws_similar1, 'result_Q2':question_similar2, 'result_A2':anws_similar2, 'result_Q3':question_similar3, 'result_A3':anws_similar3})
        #results.append(json_data)
    return json_data

if __name__ == '__main__':
    app.run()  
