#import necessary modules from hw1
from Preprocessor import text_preprocessing, get_text

#import necessary module 
import numpy as np
import pandas as pd
import math

##############1. retrieve bag of word model from training data#####################

#read training document index for each class
def get_training_index():
    '''
    training_index_txt : list of list, each element consists of document number in each class with .txt at the end
    training_index_all : list, each element consists of document number in each class
    '''
    training_index_txt = [] 
    training_index_all = []
    f = open('training_index.txt')
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        line = line[1:16]
        temp = []
        for docNum in line:
            training_index_all.append(int(docNum))
            temp.append(docNum + '.txt')
        training_index_txt.append(temp)
    return training_index_txt, training_index_all 
 
#for each class, calculate document frequency and term frequency for each term and record them by using dictionary
def calculate_df_tf(training_index_txt):
    '''
    df_list : 13 dictionary in a list, each ditionary contains word and its document frequency
    tf_list :13 dictionary in a list, each ditionary contains word and its term frequency
    '''
    df_list = []
    tf_list = []
    for index_by_class in training_index_txt:
        temp_list_df = [] #contains dictionary with keys = words, values = 1
        temp_dict_tf = dict()
        temp_dict_df = dict()
        #iterate each document in class
        for index in index_by_class:
            raw_text = get_text(index)
            text = text_preprocessing(raw_text)
            
            #count df by combining dictionaries
            temp_list_df.append(dict((word, 1) for word in set(text)))
            
            #iterate each word in document
            for word in text:
                if word not in temp_dict_tf.keys():
                    temp_dict_tf[word] = 1
                else:
                    temp_dict_tf[word] += 1
                    
        for my_dict in temp_list_df:
            for key, value in my_dict.items():
                temp_dict_df.setdefault(key, 0)
                temp_dict_df[key] += value
                
        tf_list.append(temp_dict_tf)    
        df_list.append(temp_dict_df)
    
    #do not record word:''
    for my_dict in tf_list:
        my_dict.pop('')
    
    for my_dict in df_list:
        my_dict.pop('')
        
    return df_list, tf_list

'''
check
for i in range(13):
    for word in tf_list[i].keys():
        if df_list[i][word] > tf_list[i][word]:
            print(word, tf_list[i][word], df_list[i][word])
'''
#calculate df and tf for each term for all training documents
def calculuate_df_tf_all(df_list, tf_list):
    df_all = {}
    for my_dict in df_list:
        for key, value in my_dict.items():
            df_all.setdefault(key, 0)
            df_all[key] += value
    
    tf_all = {}
    for my_dict in tf_list:
        for key, value in my_dict.items():
            tf_all.setdefault(key, 0)
            tf_all[key] += value
    
    return df_all, tf_all

'''
check
for word in df_all.keys():
    if df_all[word] > tf_all[word]:
        print(word)
'''

    
#calculate number of training document
def calculate_num_doc(training_index_txt):
    doc_num = []
    for index in training_index_txt:
        doc_num.append(len(index))
    total_training_doc_num = sum(doc_num)
    
    return total_training_doc_num

############2. for each class, choose 500 top features (words) by chi-square methods###########
def chi_square_feature_selection(df_list, df_all, total_training_doc_num):
    k = 500
    features = []
    for my_dict in df_list:
        word_list = []
        score_list = []
        for word in my_dict.keys():
            word_list.append(word)
            '''
            presentNon : number of document that this word appears in class c
            absentNon : number of document that this word doesn't appear in class c 
            presentNoff : number of document that this word appears in class other than c
            absentNoff : number of document that this word doesn't appear in class other than c
            '''
            on_topic = 15
            off_topic = total_training_doc_num - on_topic
            present = df_all[word]
            absent = total_training_doc_num - present
            
            presentNon = my_dict[word]
            absentNon = on_topic - presentNon
            presentNoff = present - presentNon
            absentNoff = absent - absentNon
            
            E_00 = absent * off_topic / total_training_doc_num
            E_01 = absent * on_topic / total_training_doc_num
            E_10 = present * off_topic / total_training_doc_num
            E_11 = present * on_topic / total_training_doc_num
    
            chi_00 = ((absentNoff - E_00) ** 2) / E_00   
            chi_01 = ((absentNon - E_01) ** 2) / E_01
            chi_10 = ((presentNoff - E_10) ** 2) / E_10
            chi_11 = ((presentNon - E_11) ** 2) / E_11
            
            score = chi_00 + chi_01 + chi_10 + chi_11
            score_list.append(score)
        score_arr = np.asarray(score_list)
        index = (-score_arr).argsort()[:k]
        feature_set = set([word_list[i] for i in index])
        features.append(feature_set)
    return features

###########3. apply NB classifier (multinomial model)##########
def train_NB_clf(tf_list, tf_all, features):
    conditional_prob = []
    count = 0
    prior = []
    #iterate each class
    for class_dict in tf_list:
        conditional_prob_dict = dict()
        prior.append(15 / 195)
        word_num = len(tf_all.keys()) ##500 * 13 or 全部不重複的字 
        sum_tf = sum(class_dict.values())
        for word in tf_all.keys():
            if word in features[count]:
                #add-one smoothing
                cond_prob = (class_dict[word] + 1) / (sum_tf + word_num)   
                conditional_prob_dict[word] = cond_prob
        conditional_prob.append(conditional_prob_dict)
        count += 1
    
    return conditional_prob, prior

def test_NB_clf(training_index_all, prior, features, conditional_prob):    
    result_class = dict()
    for index in range(1,1096,1):
        if index not in training_index_all:
            raw_text = get_text(str(index) + '.txt')
            text = text_preprocessing(raw_text)
            score_list = []
            for i in range(13):
                score = 0
                score += abs((math.log(prior[i])))
                for word in text:
                    if word in features[i]:
                        score += abs(math.log(conditional_prob[i][word]))
                score_list.append(score)
            score_arr = np.asarray(score_list)
            best_class = (-score_arr).argsort()[:1] + 1
            result_class[index] = best_class[0]
    return result_class

def save_result(result_class):
    result = pd.DataFrame(result_class.items(), columns = ['Id', 'Value'])
    result = result.set_index('Id')
    result.to_csv('hw3_result.csv')

training_index_txt, training_index_all = get_training_index()
df_list, tf_list = calculate_df_tf(training_index_txt)
df_all, tf_all = calculuate_df_tf_all(df_list, tf_list)
total_training_doc_num = calculate_num_doc(training_index_txt)
features = chi_square_feature_selection(df_list, df_all, total_training_doc_num)
conditional_prob, prior = train_NB_clf(tf_list, tf_all, features)
result_class = test_NB_clf(training_index_all, prior, features, conditional_prob)
save_result(result_class)

##############################pseudo code for feature selection and NB clf##################################
'''
SelectFeature(D, c, k)
D:training document set, c:class, k:500
#step1: calculate 500 features for each class
    chi-square = count four O and summation them 
#step2: drop other feature words in this class
return V_c (for each class, i.e. there are 13 V, 存成一個set)

NB_clf
#step 1: training
N = documentNumber
V_c for each c
V : BOW for all calss

for each c in C:
    N_c = docNumInClass
    prior = N_c / N
    
    for each t in V :
        caculate term frequency for each t
    for each t in V:
        conditionalProb = term frequency of t + 1 / summation all term frequency + V
    
    store conditionalProbability only in V_c
    return dict(c, conditionalProb for t in V_c)

#step 2: testing
for d in D:
    W = BOW from d
    score[c] = log (prior_c)
    for each c in C:
        for each t in W:
            if t in V_c:
                score[c] += conditional prob[c][t]
    return argmax_c score[c] (d文章最後被分類在哪一個class裡面)
'''