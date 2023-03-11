import numpy as np
import copy

def retrieve_index_and_tfidf_from_txt(doc_index):
    '''
    get term indexes and their unit tfidfs from specified documents
    '''
    index = []
    unit_tfidf = []
    pos = './result/doc' + str(doc_index) + '.txt'
    with open(pos) as f:
        count = 0
        for line in f.readlines():
            count+= 1
            if (count == 1 or count == 2):
                continue
            else:
                index.append(int(line.split()[0]))
                unit_tfidf.append(float(line.split()[1]))
    index = np.array(index)
    unit_tfidf = np.array(unit_tfidf)       
    return index, unit_tfidf

#14299是從pa2得來的bag-of-word model的長度
def generate_tfidf_matrix(num_doc, len_BOW = 14299):
    tfidf_matrix = np.zeros((num_doc, len_BOW))
    for doc_index in range(1, num_doc + 1, 1):
        indexes, unit_tfidf = retrieve_index_and_tfidf_from_txt(doc_index)
        count = 0
        for index in indexes:
            #print(index, unit_tfidf[count])
            tfidf_matrix[doc_index - 1][index - 1] = unit_tfidf[count]
            count += 1
    return tfidf_matrix
 
def compute_pairwise_cosine_similarity(array_1, array_2):
    cosine_similarity = np.dot(array_1, array_2)
    return cosine_similarity
    
def get_cosine_sim_matrix(num_doc, tfidf_matrix):
    cosine_sim_matrix = np.zeros((num_doc, num_doc))
    for i in range(num_doc):
        for j in range(num_doc):
            if j < i:
                similarity = compute_pairwise_cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
                cosine_sim_matrix[i][j] = similarity
                cosine_sim_matrix[j][i] = similarity
            elif j == i:
                cosine_sim_matrix[i][j] = 0
                cosine_sim_matrix[j][i] = 0
            else:
                break
    return cosine_sim_matrix   

#clustering
def get_available_cluster_list(num_doc):
    available_cluster = []
    for i in range(num_doc):
        available_cluster.append(1)
    return available_cluster

def HAC(K, num_doc, sim_matrix, tfidf_matrix, available_cluster):
    result = []
    for i in range(num_doc):
        result.append([i+1])
        
    while(sum(available_cluster) > K):
        index = np.unravel_index(np.argmax(sim_matrix, axis=None), sim_matrix.shape)
        row_index = index[0]
        column_index = index[1]
        
        result[row_index].extend(result[column_index])
        result[column_index] = 0
        available_cluster[column_index] = 0

        #update similarity with row_index
        for j in range(num_doc):
            if available_cluster[j] == 0:
                new_sim = 0
            elif (j == row_index):
                new_sim = 0
            else:
                new_sim = sim_gacc(j, row_index, tfidf_matrix, result)
            sim_matrix[j][row_index] = new_sim
            sim_matrix[row_index][j] = new_sim
            
            #change similarity that contains "column_index" to zero
            sim_matrix[j][column_index] = 0
            sim_matrix[column_index][j] = 0
    
    return result

def sim_gacc(clus_iter, clus_survive, tfidf_matrix, result):
    '''
    compute similarity for two documents by using gacc
    '''
    len_clus_iter = len(result[clus_iter])
    len_clus_survive = len(result[clus_survive])
    index_iter = np.array(result[clus_iter]) - 1
    index_survive = np.array(result[clus_survive]) - 1
    sum_iter_cluster = np.sum(tfidf_matrix[index_iter], axis = 0)
    sum_iter_survive = np.sum(tfidf_matrix[index_survive], axis = 0)
    s = np.sum([sum_iter_cluster, sum_iter_survive], axis = 0)
    similarity_gacc = (np.dot(s,s) - len_clus_iter - len_clus_survive) /  ((len_clus_iter + len_clus_survive) * (len_clus_iter + len_clus_survive - 1))
    return similarity_gacc     

def save_result(K, result):
    location = "./{}.txt".format(K)
    with open(location, 'w') as f:
        for cluster in result:
            if cluster != 0:
                cluster = sorted(cluster)
                for doc in cluster:
                    f.write("{}".format(doc))
                    f.write("\n")
                f.write("\n")
                
##########################################################################################
#compute cosine similarity for each pair of document
num_doc = 1095
tfidf_matrix = generate_tfidf_matrix(num_doc)
similarity_matrix = get_cosine_sim_matrix(num_doc, tfidf_matrix)

#clustering
K_list = [8, 13, 20]
for K in K_list:
    available_cluster = get_available_cluster_list(num_doc)
    similarity_matrix_2 = copy.copy(similarity_matrix)
    result = HAC(K, num_doc, similarity_matrix_2, tfidf_matrix, available_cluster)
    save_result(K, result)