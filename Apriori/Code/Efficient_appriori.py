def frequent_words(name , n , k):
    
    
################################ Importing required packages ################################
    
    #import sys
    #module = ["pandas" , "efficient_apriori" , "time"]
    #for modulename in module:
        #if modulename not in sys.modules:
    f = open("output_of_{}.txt".format(name),'w') # Creating output file
    try:
        import pandas as pd
        from efficient_apriori import apriori as apriori_ef
        import time
        import warnings
        warnings.filterwarnings("ignore")
        #break
    except:
        print("Install all module mentioned in documentation ")
           # print("All is imported")
    
################################ Creating link and imporing data ############################
    
    link = ["http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab." + str(name) + ".txt", "http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword." + str(name) + ".txt.gz"]
   
    try:
        vocab = pd.read_csv(link[0], header = None)
        docword = pd.read_csv(link[1], header = None)
    except:
        print("Check your internet connection", file = f)# Writing in file : When internet is slow
        return("Check your internet connection")
    
################################ Starting clock #############################################
    
    start_time = time.clock()
    
################################ Prepairing data for apriori ################################
    
    data = docword[:3]
    
    data_NNZ = docword[3:]
    data_NNZ.columns = ["NNZ"]
    
    data_NNZ = data_NNZ['NNZ'].str.split(" ")
    
    data_NNZ = pd.DataFrame(list(data_NNZ))
    data_NNZ.columns = ["Doc_id","Word_id","Count"]
    
        
    dict = data_NNZ.groupby('Doc_id')['Word_id'].apply(list).to_dict()

    words_in_doc = list(dict.values())
    
##################################### Apriori#### ###########################################
    
    itemsets, rules = apriori_ef(words_in_doc, min_support = n,  min_confidence=1)
    
    try:
        frequent_word_id = list(itemsets[k].keys())
    except KeyError:
        print("No such values", file = f)# Writing in file : When no itemset of length is available
        return("No such values")
    
################################# Mapping word ids to word ##################################
    
    frequent_word = []
    for i in range(len(frequent_word_id)):
        temp_word = []
        for j in range(k):
            temp_word.append(vocab[0][int(frequent_word_id[i][j])-1])
        frequent_word.append(temp_word)
        
    frequent_word_with_len = pd.DataFrame(frequent_word, columns = list("word "+str(i+1) for i in range(k))) 
    frequent_word_with_len["Freq"] = list(itemsets[k].values())

################################ Ending clock ###############################################
        
    end_time = time.clock()
    time_apriori = (end_time-start_time)    
    
    print("Time taken is ", time_apriori ,file = f) # Writing in file : Time taken
    print("\n", file = f)                           # Writing in file :
    print("Used minimum support provided by the user is {} and corrosponding min freq is {}".format(n, int(data[0][0])*n ), file = f)# Writing in file : Minimum support and length 
    print("\n", file = f)                           # Writing in file 
    print("Frequent words of length {} are ".format(k) , file = f) # Writing in file 
    print("\n", file = f)                           # Writing in file
    print(frequent_word_with_len, file = f)         # Writing in file : Output
    
    return(frequent_word, time_apriori)
    
     