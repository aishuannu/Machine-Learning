def frequent_words(link , n , k):
    
    try:
        import pandas as pd
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori
        import time
        import warnings
        warnings.filterwarnings("ignore")
        #break
    except:
        print("Install all module mentioned in documentation ")

    
    
    #vopcab = pd.read_csv("link[0]", header = None)
    docword = pd.read_csv("link[1]", header = None)
    
    #data = docword[0:3]
    data_NNZ = docword[3:]
    data_NNZ.columns = ["NNZ"]
    
    data_NNZ = data_NNZ['NNZ'].str.split(" ")
    
    data_NNZ = pd.DataFrame(list(data_NNZ))
    data_NNZ.columns = ["Doc_id","Word_id","Count"]
    
    dict = data_NNZ.groupby('Doc_id')['Word_id'].apply(list).to_dict()
    words_in_doc = list(dict.values())
    
    start_time = time.clock() #starting Time
    
    te = TransactionEncoder()
    te_ary = te.fit(words_in_doc).transform(words_in_doc)
    sparse_words_matrix = pd.SparseDataFrame(te_ary, columns=te.columns_, default_fill_value=False)
    
    frequent_itemsets = apriori(sparse_words_matrix, min_support = n , max_len=3 ,use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    
    end_time = time.clock()
    time_apriori = (end_time-start_time)/60
    
    final_call = frequent_itemsets[frequent_itemsets.length == k].itemsets
    return(final_call, time_apriori)
    
    
    
    
    