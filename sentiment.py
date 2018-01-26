#/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import logging
import pandas as pd
import numpy as np
import nltk.data
import pickle
import gensim
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn import naive_bayes, svm, preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.grid_search import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection.univariate_selection import chi2, SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


##################### Function Definition #####################

def evaluate_on_testdata(test_vec , svc , ground_truth):
    pred = svc.predict(test_vec)
    print accuracy_score(ground_truth , pred)

def clean_review(raw_review, language, remove_stopwords = False, output_format = "string" ):
    """
    Input:
            raw_review: raw text of a movie review
            remove_stopwords: a boolean variable to indicate whether to remove stop words
            output_format: if "string", return a cleaned string 
                           if "list", a list of words extracted from cleaned string.
    Output:
            Cleaned string or list.
    """
    
    # Remove HTML markup
    #text = BeautifulSoup(raw_review)
    text = BeautifulSoup(raw_review , 'lxml')
    
    # Keep only characters
    text = re.sub("[^a-zA-Z]", " ", text.get_text())
    
    # Split words and store to list
    text = text.lower().split()
    
    if remove_stopwords:
    
        # Use set as it has O(1) lookup time
        stops = set(stopwords.words(language))
        words = [w for w in text if w not in stops]
    
    else:
        words = text
    
    # Return a cleaned string or list
    if output_format == "string":
        return " ".join(words)
        
    elif output_format == "list":
        return words
    
    
def review_to_doublelist(review, language, tokenizer, remove_stopwords = False ):
    """
    Function which generates a list of lists of words from a review for word2vec uses.
    
    Input:
        review: raw text of a movie review
        tokenizer: tokenizer for sentence parsing
                   nltk.data.load('tokenizers/punkt/english.pickle')
        remove_stopwords: a boolean variable to indicate whether to remove stop words
    
    Output:
        A list of lists.
        The outer list consists of all sentences in a review.
        The inner list consists of all words in a sentence.
    """
    
    # Create a list of sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    sentence_list = []
    
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentence_list.append(clean_review(raw_sentence, language, False, "list" ))         
    return sentence_list


def review_to_vec(words, model, num_features , index2word_set):
    """
    Function which generates a feature vector for the given review.
    
    Input:
        words: a list of words extracted from a review
        model: trained word2vec model
        num_features: dimension of word2vec vectors
        
    Output:
        a numpy array representing the review
    """
    
    feature_vec = np.zeros((num_features), dtype="float32")
    word_count = 0
    
    
    
    for word in words:
        if word in index2word_set: 
            word_count += 1
            feature_vec += model[word]

    if word_count == 0:
        word_count = 1

    feature_vec /= word_count

    return feature_vec
    
    
def gen_review_vecs(reviews, model, num_features):
    """
    Function which generates a m-by-n numpy array from all reviews,
    where m is len(reviews), and n is num_feature
    
    Input:
            reviews: a list of lists. 
                     Inner lists are words from each review.
                     Outer lists consist of all reviews
            model: trained word2vec model
            num_feature: dimension of word2vec vectors
    Output: m-by-n numpy array, where m is len(review) and n is num_feature
    """

    curr_index = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")

    # index2word is a list consisting of all words in the vocabulary
    # Convert list to set for speed
    index2word_set = set(model.wv.index2word)
    for review in reviews:

       #if curr_index%1000 == 0.:
       #    print "Vectorizing review %d of %d" % (curr_index, len(reviews))
   
       review_feature_vecs[curr_index] = review_to_vec(review, model, num_features , index2word_set)
       curr_index += 1
       
    return review_feature_vecs
    
    
##################### End of Function Definition #####################

def main():

    os.chdir("/Users/yingbei.tong@ibm.com/Desktop/workspace/sentiment.analysis")


    ##################### Initialization #####################

    write_to_csv = False
    tune_parameter = False
    Mix = True

    # term_vector_type = {"TFIDF", "Binary", "Int", "Word2vec", "Word2vec_pretrained"}
    # {"TFIDF", "Int", "Binary"}: Bag-of-words model with {tf-idf, word counts, presence/absence} representation
    # {"Word2vec", "Word2vec_pretrained"}: Google word2vec representation {without, with} pre-trained models
    # Specify model_name if there's a pre-trained model to be loaded
    #vector_type = "TFIDF"
    vector_type = 'Word2vec_pretrained'

    #model_name = "selftrainBad.bin"

    model_name = "wiki.fr.vec"


    # model_type = {"bin", "reg"}
    # Specify whether pre-trained word2vec model is binary
    #model_type = "bin"
       
    # Parameters for word2vec
    # num_features need to be identical with the pre-trained model
    num_features = 300    # Word vector dimensionality                      
    min_word_count = 5   # Minimum word count to be included for training                      
    num_workers = 4       # Number of threads to run in parallel
    context = 4         # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words

    # training_model = {"RF", "NB", "SVM", "BT", "no"}
    training_model = "SVM"

    # feature scaling = {"standard", "signed", "unsigned", "no"}
    # Note: Scaling is needed for SVM
    scaling = "no"

    # dimension reduction = {"SVD", "chi2", "no"}
    # Note: For NB models, we cannot perform truncated SVD as it will make input negative
    # chi2 is the feature selectioin based on chi2 independence test
    dim_reduce = "no"
    num_dim = 200

    ##################### End of Initialization #####################

    print('parameter settings: ')
    print('vector_type:' + vector_type)
    print('training_model: ' + training_model)
    print('scaling: ' + scaling)
    print('dim_reduce: ' + dim_reduce )

    ########################### Main Program ###########################

    train_list = []
    test_list_t = []
    test_list_h = []
    test_list_c = []
    word2vec_input = []
    train_list2 = []
    pred = []

    language = 'french'

    train_language = 'german'
    test_language = 'french'

    trainFile = train_language + 'TrainData_100k.csv'
    trainFile2 = test_language + 'TrainData_100k.csv' ##

    testFile_t = test_language + 'TestData_cftwt.csv'
    testFile_h = test_language + 'TestData_cfdata.csv'
    testFile_c = test_language + 'TestData_deft.csv'
    #unlabFile = 'frenchUnlab.csv'

    train_data = pd.read_csv("data/" + trainFile, header=0, delimiter=",", quoting=0 )#, encoding='utf-8')
    if Mix == True:
        train_data2 = pd.read_csv("data/" + trainFile2, header=0, delimiter=",", quoting=0 )

    test_data_t = pd.read_csv("data/" + testFile_t, header=0, delimiter=",", quoting=0)# , encoding='utf-8')
    test_data_h = pd.read_csv("data/" + testFile_h, header=0, delimiter=",", quoting=0)# , encoding='utf-8')
    test_data_c = pd.read_csv("data/" + testFile_c, header=0, delimiter=",", quoting=0)# , encoding='utf-8')
   # unlab_train_data = pd.read_csv("data/" + unlabFile, header=0, delimiter=",", quoting=0)# , encoding='utf-8')


    if vector_type == "Word2vec":
        unlab_train_data = pd.read_csv("data/frenchUnlabeledTrainData.csv", header=0, delimiter=",", quoting=0)
        tokenizer = nltk.data.load('tokenizers/punkt/'+ language+'.pickle')
        logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

    ground_truth_t = test_data_t.sentiment
    ground_truth_h = test_data_h.sentiment
    ground_truth_c = test_data_c.sentiment
    # Extract words from reviews
    # xrange is faster when iterating
    if vector_type == "Word2vec" or vector_type == "Word2vec_pretrained":
        
        for i in xrange(0, len(train_data.review)):
            
            if vector_type == "Word2vec":
                # Decode utf-8 coding first
                word2vec_input.extend(review_to_doublelist(train_data.review[i].decode("utf-8"), language, tokenizer ))
                
           # print train_data.id[i]
            train_list.append(clean_review(train_data.review[i], language, output_format="list" ))
            #if i%1000 == 0:
                #print "Cleaning training review", i

        if Mix == True:
            for i in xrange(0, len(train_data2.review)):
                        
               # print train_data.id[i]
                train_list2.append(clean_review(train_data2.review[i], language, output_format="list" ))
                #if i%1000 == 0:
                    #print "Cleaning training review", i

           
        if vector_type == "Word2vec":                
            for i in xrange(0, len(unlab_train_data.review)):
                #print unlab_train_data.review[i]
                word2vec_input.extend(review_to_doublelist(unlab_train_data.review[i].decode("utf-8"), language, tokenizer))
                #if i%1000 == 0:
                    #print "Cleaning unlabeled training review", i
        
        for i in xrange(0, len(test_data_t.review)):
            test_list_t.append(clean_review(test_data_t.review[i], language, output_format="list"))
            #if i%1000 == 0:
                #print "Cleaning test review", i  
        for i in xrange(0, len(test_data_h.review)):
            test_list_h.append(clean_review(test_data_h.review[i], language, output_format="list"))
            #if i%1000 == 0:
                #print "Cleaning test review", i   
        for i in xrange(0, len(test_data_c.review)):
            test_list_c.append(clean_review(test_data_c.review[i], language, output_format="list"))
            #if i%1000 == 0:
                #print "Cleaning test review", i        

    elif vector_type != "no": 
        for i in xrange(0, len(train_data.review)):
            
            # Append raw texts rather than lists as Count/TFIDF vectorizers take raw texts as inputs
            train_list.append(clean_review(train_data.review[i], language) )
            #if i%1000 == 0:
               # print "Cleaning training review", i

        for i in xrange(0, len(test_data.review)):
            
            # Append raw texts rather than lists as Count/TFIDF vectorizers take raw texts as inputs
            test_list.append(clean_review(test_data.review[i], language))
            #if i%1000 == 0:
            #    print "Cleaning test review", i


    # Generate vectors from words
    if vector_type == "Word2vec_pretrained" or vector_type == "Word2vec":
        
        if vector_type == "Word2vec_pretrained":
            print "Loading the pre-trained model"
            if model_name.endswith == ".bin":
                #model = word2vec.Word2Vec.load_word2vec_format(model_name, binary=True)
                model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True , unicode_errors='ignore')
            else:
                #model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=False , unicode_errors='ignore') 
                train_model = gensim.models.KeyedVectors.load_word2vec_format('wiki.multi.'+ train_language +'.vec', binary=False , unicode_errors='ignore') 
                test_model = gensim.models.KeyedVectors.load_word2vec_format('wiki.multi.'+ test_language +'.vec', binary=False , unicode_errors='ignore') 

        if vector_type == "Word2vec":
            print "Training word2vec word vectors"
            model = word2vec.Word2Vec(word2vec_input, workers=num_workers, \
                                    size=num_features, min_count = min_word_count, \
                                    window = context, sample = downsampling)
        
            # If no further training and only query is needed, this trims unnecessary memory
            model.init_sims(replace=True)
        
            # Save the model for later use
            word_vectors = model.wv
            model.save(model_name)
        
        print "Vectorizing training review"
        train_vec = gen_review_vecs(train_list, train_model, num_features)
        if Mix == True:
            train_vec2 = gen_review_vecs(train_list2, test_model, num_features)
            train_vec = np.append(train_vec , train_vec2 , axis = 0)
            #train_vec = np.concatenate((train_vec, train_vec2) , axis = 0)

        print "Vectorizing test review"
        test_vec_c = gen_review_vecs(test_list_c,test_model, num_features)
        test_vec_h = gen_review_vecs(test_list_h,test_model, num_features)
        test_vec_t = gen_review_vecs(test_list_t,test_model, num_features)
        
        
    elif vector_type != "no": 
        if vector_type == "TFIDF":
            # Unit of gram is "word", only top 5000/10000 words are extracted
            count_vec = TfidfVectorizer(analyzer="word", max_features=10000, ngram_range=(1,2), sublinear_tf=True)
            
        elif vector_type == "Binary" or vector_type == "Int":       
            count_vec = CountVectorizer(analyzer="word", max_features=10000, \
                                        binary = (vector_type == "Binary"), \
                                        ngram_range=(1,2))
        
        # Return a scipy sparse term-document matrix
        print "Vectorizing input texts"
        train_vec = count_vec.fit_transform(train_list)
        test_vec_h = count_vec.transform(test_list_h)
        test_vec_t = count_vec.transform(test_list_t)
        test_vec_c = count_vec.transform(test_list_c)


    # Dimemsion Reduction
    if dim_reduce == "SVD":
        print "Performing dimension reduction"
        svd = TruncatedSVD(n_components = num_dim)
        train_vec = svd.fit_transform(train_vec)
        test_vec_h = svd.transform(test_vec_h)
        test_vec_t = svd.transform(test_vec_t)
        test_vec_c = svd.transform(test_vec_c)
        print "Explained variance ratio =", svd.explained_variance_ratio_.sum()

    elif dim_reduce == "chi2":
        print "Performing feature selection based on chi2 independence test"
        fselect = SelectKBest(chi2, k=num_dim)
        train_vec = fselect.fit_transform(train_vec, train_data.sentiment)
        test_vec = fselect.transform(test_vec)

    # Transform into numpy arrays
    if "numpy.ndarray" not in str(type(train_vec)):
        train_vec = train_vec.toarray()
        test_vec_h = test_vec_h.toarray()  
        test_vec_t = test_vec_t.toarray()  
        test_vec_c = test_vec_c.toarray()  


    # Feature Scaling
    if scaling != "no":

        if scaling == "standard":
            scaler = preprocessing.StandardScaler()
        else: 
            if scaling == "unsigned":
                scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
            elif scaling == "signed":
                scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        
        print "Scaling vectors"
        train_vec = scaler.fit_transform(train_vec)
        test_vec = scaler.transform(test_vec)
        
        
    # Model training 
    if training_model == "RF" or training_model == "BT":
        
        # Initialize the Random Forest or bagged tree based the model chosen
        rfc = RFC(n_estimators = 100, oob_score = True, \
                  max_features = (None if training_model=="BT" else "auto"))
        print "Training %s" % ("Random Forest" if training_model=="RF" else "bagged tree")
        rfc = rfc.fit(train_vec, train_data.sentiment)
        print "OOB Score =", rfc.oob_score_
        pred = rfc.predict(test_vec)
        
    elif training_model == "NB":
        nb = naive_bayes.MultinomialNB()
        cv_score = cross_val_score(nb, train_vec, train_data.sentiment, cv=10)
        print "Training Naive Bayes"
        print "CV Score = ", cv_score.mean()
        nb = nb.fit(train_vec, train_data.sentiment)
        pred = nb.predict(test_vec)
        
    elif training_model == "SVM":
        svc = svm.LinearSVC()
        #svc = svm.SVC(kernel = 'linear', probability = True) #seems it takes so long time to train??
        print 'complete 0'
        param = {'C': [1e15,1e13,1e11,1e9,1e7,1e5,1e3,1e1,1e-1,1e-3,1e-5]}
        print "Training SVM"

        

        if tune_parameter == True:
            svc = GridSearchCV(estimator=svc, param_grid = param, cv=10)

        #next 2 Lines are for enable probability
        svc = CalibratedClassifierCV(svc)

        #print 'complete 1'

        sentiment_array = []
        for sent in train_data.sentiment:
            sentiment_array.append(sent)
        if Mix == True:
            for sent in train_data2.sentiment:
                sentiment_array.append(sent)

        svc = svc.fit(train_vec, sentiment_array)
        #svc = svc.fit(train_vec, train_data.sentiment)

        print 'complete 2'
        #pred_t = svc.predict(test_vec_t)
        #pred_h = svc.predict(test_vec_h)
        #pred_c = svc.predict(test_vec_c)

        #pred_proba_t = svc.predict_proba(test_vec_t)

        #pred1 = svc.predict_proba(test_vec)
        #print(pred1)
        #print(pred_proba_t)
        print('Accuracy on "cftwt.csv" dataset:')
        evaluate_on_testdata(test_vec_t, svc , ground_truth_t)
        print('Accuracy on "cfdata.csv" dataset:')
        evaluate_on_testdata(test_vec_h, svc , ground_truth_h)
        print('Accuracy on "deft.csv" dataset:')
        evaluate_on_testdata(test_vec_c, svc , ground_truth_c)
        print('training dataset is : ')
        if Mix:
            print "used Mixed datasets"
        print trainFile

        if tune_parameter == True:
            print "Optimized parameters:", svc.best_estimator_ #print the best parameter when using GridSearchCV
            print "Best CV score:", svc.best_score_

        #filename =vector_type+ 'finalized_model.pkl'
        #s = pickle.dump(svc, open(filename, 'wb'))
        
    # Output the results
    if write_to_csv:
        output = pd.DataFrame(data = {"id": test_data.id, "sentiment": pred})
        output.to_csv("data/" + vector_type +"submission.csv", index=False)
        

if __name__ == "__main__":
    main()










    

