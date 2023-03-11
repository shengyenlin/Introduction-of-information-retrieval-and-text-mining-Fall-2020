from nltk.stem.porter import PorterStemmer

######### hw1 ############
def get_text(position):
    '''
    read content from txt
    '''
    f = open("./IRTM/" + position)
    text = []
    for line in f.readlines():
        text.append(line)
    f.close()
    return text

def tokenize_text(text):
    '''
    去除換行符號、切開並將一些字還原
    '''
    replace_dictionary = {"'s": " is", "'ve": " have", "'re": " are", "'t'":' not', '/':' ','.':' '}
    #replaces = [',', '/']
    text_tokenized = []
    for sentence in text:
        sentence = sentence.strip()
        for key, value in replace_dictionary.items():
            sentence = sentence.replace(key, value)
        words = sentence.split(' ')
        for word in words:
            text_tokenized.append(word)
    return text_tokenized 
      
def delete_delimiters(text_tokenized):
    '''
    刪除標點符號
    '''
    delimiter = ['.', '?','-','`','!','\"','\'','(',')','（','）','$',':',';','*','','%','&','_','{','}','#','@',',']
    text_final = []
    for token in text_tokenized:
        word = ""
        for element in token:
            if element not in delimiter:
                word += element
        word = word.lower()
        text_final.append(word)
    return text_final

def lowercase_words(text_tokenized):
    '''
    將字母全部縮成小寫
    '''
    text_tokenized = [word.lower() for word in text_tokenized]
    return text_tokenized

def stemming_word(text_tokenized):
    '''
    Stemming using Porter’s algorithm
    '''
    porter_stemmer = PorterStemmer()
    text_tokenized = [porter_stemmer.stem(word) for word in text_tokenized]
    return text_tokenized
    
def remove_stopwords(text_tokenized):
    '''
    Stopword removal
    '''
    f = open('stopwords.txt')
    stopwords = []
    for stopword in f.readlines():
        stopwords.append(stopword.strip())
    f.close()
    text_final = [word for word in text_tokenized if word not in stopwords]
    return text_final

def text_preprocessing(text):
    text_tokenized = tokenize_text(text)
    text_no_del = delete_delimiters(text_tokenized)
    text_lower = lowercase_words(text_no_del)
    text_stemmed = stemming_word(text_lower)
    text_final = remove_stopwords(text_stemmed)
    return text_final

######### hw1 ############