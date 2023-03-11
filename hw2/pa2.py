#import necessary modules from hw1
from Preprocessor import text_preprocessing, get_text

#import necessary module 
import numpy as np

###############################algos##############################################    
#####Task 1: caculate document frequency#####
def generate_txt_list():
    '''
    generate a list containing ['1.txt', '2.txt', ...]
    '''
    number_txt = []
    for i in range(1095):
        number_txt.append(str(i+1) + ".txt")
    return number_txt

def get_raw_text(number_txt):
    '''
    retrieve 1095 documents from txt files
    output : list of list ([news[preprocessed words, prerpocessed words, ...]])
    '''
    raw_text = []
    for txt in number_txt:
        text = get_text(txt)
        text_final = text_preprocessing(text)
        raw_text.append(text_final)
    return raw_text

def get_BOW(raw_text):
    '''
    generate bag of words model
    '''
    bag_of_words = []
    for text in raw_text:
        bag_of_words.extend(text)
    bag_of_words = dict.fromkeys(bag_of_words)
    bag_of_words = sorted(bag_of_words)
    bag_of_words = bag_of_words[1:]
    return bag_of_words

def get_df(bag_of_words, raw_text):
    '''
    generate document frequency in a list
    the order of the words is unchanged
    '''
    df = []
    for word_find in bag_of_words:
        count = 0
        for news in raw_text:
            for word_news in news:
                if (word_find == word_news):
                    count += 1
                    break
        df.append(count)
    return df
    
def get_word_index(bag_of_words):
    '''
    generate index for each word in BOW
    '''
    words_index = [i+1 for i in range(len(bag_of_words))]
    return words_index

def save_df(words_index, bag_of_words, df):
    '''
    save document frequency of each word into dictionary.txt
    '''
    with open('dictionary.txt', 'w') as f:
        f.write('{:<15} {:<15} {}'.format('t_index', 'term', 'df'))
        f.write('\n')
        for i in range(len(words_index)):
            f.write('{:<15} {:<15} {}'.format(words_index[i], bag_of_words[i], df[i]))
            f.write('\n')

#####Task 2: Transfer each document into a tf-idf unit vector#####
def get_doc_length(raw_text):
    '''
    return number of total documents
    '''
    num_doc = len(raw_text)
    return num_doc

def get_BOW_length(bag_of_words):
    '''
    return length of BOW
    '''
    len_BOW = len(bag_of_words)
    return len_BOW

def get_idf_array(df):
    '''
    turn df into idf numpy array
    '''
    df_array = np.array(df)
    num_doc = len(raw_text)
    idf_array = np.log10(num_doc/df_array)
    return idf_array

def get_tf_matrix(raw_text, bag_of_words):
    '''
    building tf matrix (14269, 1095)
    row : term, column : document
    '''
    tf = []
    #1095, 14299
    for news in raw_text:
        temp = []
        for word_find in bag_of_words:
            count = 0
            for word_news in news:
                if (word_find == word_news):
                    count += 1
            temp.append(count)
        tf.append(temp)
    #tf-idf
    tf_matrix = np.array(tf)
    tf_matrix = tf_matrix.transpose()
    return tf_matrix

def get_tfidf_matrix(tf_matrix, idf_array, len_BOW):
    '''
    combine tf matrix and idf vector into tfidf matrix 
    return (14269, 1095) (term, document)
    '''
    tfidf_list = []
    for i in range(len_BOW):
        tfidf = tf_matrix[i] * idf_array[i]
        tfidf_list.append(tfidf)
    tfidf_matrix = np.array(tfidf_list)
    return tfidf_matrix

def get_unit_tfidf_matrix(num_doc, tfidf_matrix):
    '''
    normalize tfidf matrix with respect to each document
    return (14269, 1095) (term, document)
    '''
    unit_tfidf_list = []
    for i in range(num_doc):
        tfidf_sum = np.sqrt(np.sum(np.square(tfidf_matrix.transpose()[i])))
        unit_tfidf_list.append(tfidf_matrix.transpose()[i]/tfidf_sum)
    unit_tfidf_matrix = np.array(unit_tfidf_list)
    unit_tfidf_matrix = unit_tfidf_matrix.transpose()
    return unit_tfidf_matrix

def save_unit_tfidf_matrix(num_doc, unit_tfidf_matrix, number_txt):
    '''
    save tf-idf file
    '''
    for i in range(num_doc):
        doc_tfidf = unit_tfidf_matrix.transpose()[i]
        #nonzero_index start from zero to 14298
        nonzero_index = np.nonzero(doc_tfidf)[0]
        location = './result/doc'+ number_txt[i]
        with open(location, 'w') as f:
            f.write('{}'.format(len(nonzero_index)))
            f.write('\n')
            f.write('{:<10} {}'.format('t_index', 'tf-idf'))
            f.write('\n')
            for index in nonzero_index:
                f.write('{:<10} {}'.format(words_index[index], doc_tfidf[index]))
                f.write('\n')

#####Task 3: compute cosine similarity#####
def retrieve_index_and_tfidf(num_doc1, num_doc2):
    '''
    get term indexes and their unit tfidfs from specified documents
    '''
    index_1 = []
    index_2 = []
    unit_tfidf_1 = []
    unit_tfidf_2 = []
    pos1 = './result/doc' + str(num_doc1) + '.txt'
    pos2 = './result/doc' + str(num_doc2) + '.txt'
    with open(pos1) as f:
        count = 0
        for line in f.readlines():
            count+= 1
            if (count == 1 or count == 2):
                continue
            else:
                index_1.append(int(line.split()[0]))
                unit_tfidf_1.append(float(line.split()[1]))
    with open(pos2) as f:
        count = 0
        for line in f.readlines():
            count+= 1
            if (count == 1 or count == 2):
                continue
            else:
                index_2.append(int(line.split()[0]))
                unit_tfidf_2.append(float(line.split()[1]))
    index_1 = np.array(index_1)
    index_2 = np.array(index_2)
    unit_tfidf_1 = np.array(unit_tfidf_1)
    unit_tfidf_2 = np.array(unit_tfidf_2)           
    return index_1, index_2, unit_tfidf_1, unit_tfidf_2

def compute_cosine_similarity(index_1, index_2, unit_tfidf_1, unit_tfidf_2, len_BOW):
    unit_tfidf_array_1 = np.zeros(len_BOW)
    unit_tfidf_array_2 = np.zeros(len_BOW)
    count_1 = 0
    for index in index_1:
        unit_tfidf_array_1[index-1] = unit_tfidf_1[count_1]
        count_1 += 1 
        
    count_2 = 0
    for index in index_2:
        unit_tfidf_array_2[index-1] = unit_tfidf_2[count_2]
        count_2 += 1 
    cosine_similarity = np.dot(unit_tfidf_array_1, unit_tfidf_array_2)
    return cosine_similarity

###############################algos##############################################    

#####Task 1: get document frequency#####
number_txt = generate_txt_list()
raw_text = get_raw_text(number_txt)
bag_of_words = get_BOW(raw_text)
df = get_df(bag_of_words, raw_text)
words_index = get_word_index(bag_of_words)
save_df(words_index, bag_of_words, df)
#195秒完成

#####Task 2: Transfer each document into a tf-idf unit vector#####
num_doc = get_doc_length(raw_text)
len_BOW = get_BOW_length(bag_of_words)
idf_array = get_idf_array(df)
tf_matrix = get_tf_matrix(raw_text, bag_of_words)
tfidf_matrix = get_tfidf_matrix(tf_matrix, idf_array, len_BOW)
unit_tfidf_matrix = get_unit_tfidf_matrix(num_doc, tfidf_matrix)
save_unit_tfidf_matrix(num_doc, unit_tfidf_matrix, number_txt)
#140秒完成

#####Task 3: compute cosine similarity#####
num_doc1 = 1
num_doc2 = 2
len_BOW = get_BOW_length(bag_of_words)
index_1, index_2, unit_tfidf_1, unit_tfidf_2 = retrieve_index_and_tfidf(num_doc1, num_doc2)
cosine_similarity = compute_cosine_similarity(index_1, index_2, unit_tfidf_1, unit_tfidf_2, len_BOW)
print(f"Cosine similarity of document{num_doc1} and document{num_doc2} is", round(cosine_similarity, 5))

