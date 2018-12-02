# -*- coding: utf-8 -*-
##------------------------------------------------------------------------

#Article content based document vetor approach (spacy document vector)

##------------------------------------------------------------------------
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.externals import joblib
import collections
import scipy.spatial

### read the data set 

articles_df = pd.read_csv('ArticleSharing/shared_articles.csv')
###------update data types of article dataframe
articles_df = articles_df.astype({"timestamp": int, "contentId": int,"authorPersonId":int,"authorSessionId":int})

articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED'] ## only content shared considered
articles_df.head(5)

users_interactions=pd.read_csv('ArticleSharing/users_interactions.csv')
users_interactions=users_interactions.astype({"timestamp": int, "contentId": int,"personId":int,"sessionId":int})


## allow weights to actions

event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 3.0, 
   'FOLLOW': 4.0,
   'COMMENT CREATED': 5.0,  
}

users_interactions['eventStrength'] = users_interactions['eventType'].apply(lambda x: event_type_strength[x])

## handling cold-start, consider user with atleast 5 interactuons
users_interactions_count_df = users_interactions.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))


      
print('# of interactions: %d' % len(users_interactions))
interactions_from_selected_users_df = users_interactions.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'personId',
               right_on = 'personId')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))


####  accumulate all interaction scores  &  smooth the overall interation by taking log      
def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = interactions_from_selected_users_df \
                    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                    .apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)  


#### test train divide the data set:

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['personId'], 
                                   test_size=0.20,
                                   random_state=20
                                   )

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))
      
      
interactions_train_df=interactions_train_df.astype({"contentId": int,"personId":int})
interactions_test_df=interactions_test_df.astype({"contentId": int,"personId":int})




      
### content based recommendation

##  tf idf vectorize

stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords


vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=4000,
                     stop_words=stopwords_list)

item_ids = articles_df['contentId'].tolist()

article_language = articles_df['lang'].tolist()

##analyse for title  1 time 2 time 3 times 
tfidf_matrix = vectorizer.fit_transform(articles_df['url'] + " " +articles_df['title'] + " " +articles_df['url'] + " " +articles_df['title'] + " " + articles_df['text'])
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix=tfidf_matrix.todense()



#################################CREATE the document vector#########################
from nltk import word_tokenize


def stopWordRemoval(text,language):
    if language=='en':
        stopWordList=set(stopwords.words('english'))
    else:
        stopWordList=set(stopwords.words('portuguese'))
    return ' '.join(i for i in word_tokenize(text.lower()) if i not in stopWordList)
    
    
import spacy
nlp_english=spacy.load('en')
nlp_portugese=spacy.load('pt')

if 1==1:## working with document vector   
    featureMatrix=[]

    index=-1
    AcrticleContentmatrix=articles_df['title'] + " " + articles_df['text']
        
    for doc in AcrticleContentmatrix:
        index+=1
        doc=stopWordRemoval(doc,article_language[index])
        
        if len(featureMatrix)==0:
            try:
                if article_language[index]=='en':
                    featureMatrix=nlp_english(doc).vector
                elif article_language[index]=='pt':
                    print("portugese",doc)
                    featureMatrix=nlp_portugese(doc).vector                    
            except:
                print("Exception in ",doc)

        else:
            try:
                if article_language[index]=='en':
                    featureMatrix=np.vstack((featureMatrix,nlp_english(doc).vector))
                elif article_language[index]=='pt':
                    print("portugese",doc)
                    featureMatrix=np.vstack((featureMatrix,nlp_portugese(doc).vector))
            except:
                print("Exception in ",doc)
                
    tfidf_matrix=featureMatrix        

####################################################################################

      
### create user profile matrix for the test set

dict_trainset_interactions_of_a_user={}
dict_testset_interactions_of_a_user={}
dict_article_details_based_on_acticle_id={}
dict_user_details_based_user_id={}

## read the interaction details-----------------------------------------------------
## train set interaction of a user--------------------------------

interactionDetails=interactions_train_df.astype(str).values
interactionDetails[:,0]=interactionDetails[:,0].astype(int)
interactionDetails[:,1]=interactionDetails[:,1].astype(int)
interactionDetails[:,2]=interactionDetails[:,2].astype(float)


for index_ in range(len(interactionDetails)):
    if interactionDetails[index_][0] not in dict_trainset_interactions_of_a_user.keys():
        dict_trainset_interactions_of_a_user[interactionDetails[index_][0]]=[(interactionDetails[index_][1],interactionDetails[index_][2])]
    else:
        dict_trainset_interactions_of_a_user[interactionDetails[index_][0]].append((interactionDetails[index_][1],interactionDetails[index_][2]))
        
## test set interaction of a user--------------------------------
test_interactionDetails=interactions_test_df.astype(str).values
test_interactionDetails[:,0]=test_interactionDetails[:,0].astype(int)
test_interactionDetails[:,1]=test_interactionDetails[:,1].astype(int)
test_interactionDetails[:,2]=test_interactionDetails[:,2].astype(float)

for index_ in range(len(test_interactionDetails)):
    if test_interactionDetails[index_][0] not in dict_testset_interactions_of_a_user.keys():
        dict_testset_interactions_of_a_user[test_interactionDetails[index_][0]]=[(test_interactionDetails[index_][1],test_interactionDetails[index_][2])]
    else:
        dict_testset_interactions_of_a_user[test_interactionDetails[index_][0]].append((test_interactionDetails[index_][1],test_interactionDetails[index_][2]))




## find the set of articles the user has reviewed in test set
dict_test_set_user_article={}
for i in  dict_testset_interactions_of_a_user.keys():
       for eachValues in dict_testset_interactions_of_a_user[i]:
           
           if i not in dict_test_set_user_article.keys():
               dict_test_set_user_article[i]=[eachValues[0]]
           else:
               if eachValues[0] not in dict_test_set_user_article[i]:
                   dict_test_set_user_article[i].append(eachValues[0])

## find the set of articles the user has reviewed in train set
dict_train_set_user_article={}
for i in  dict_trainset_interactions_of_a_user.keys():
       for eachValues in dict_trainset_interactions_of_a_user[i]:
           
           if i not in dict_train_set_user_article.keys():
               dict_train_set_user_article[i]=[eachValues[0]]
           else:
               if eachValues[0] not in dict_train_set_user_article[i]:
                   dict_train_set_user_article[i].append(eachValues[0])


## read the article details-----------------------------------------------------
articlesDetails=np.array(articles_df.values)
for index_ in range(len(articlesDetails)):
    dict_article_details_based_on_acticle_id[articlesDetails[index_][2]]=articlesDetails[index_]
## read the user details-----------------------------------------------------
userDetails=interactions_from_selected_users_df.values
for index_ in range(len(userDetails)):
    dict_user_details_based_user_id[userDetails[index_][3]]=userDetails[index_]

##-------------------------------------------------------------------------------------
   
def getArticleLanguageofAnArticle(articleID):
   return dict_article_details_based_on_acticle_id[articleID][12]
    
### ############################# '''build user matric from the articles he has interacted'''
articleIDList= list(articlesDetails[:,2])
userIDFeatures={}  

def getArticleTextBasedOnArticleID(articleID_):
    try:
        text_= dict_article_details_based_on_acticle_id[articleID_][11]
    except:
         text_=""
    return text_   
    

for key_ in dict_trainset_interactions_of_a_user.keys():
#    print("--------  working for user",key_,"")
    tempFeature=[]
    text=""
    articleTouples=dict_trainset_interactions_of_a_user[key_]
    for article in articleTouples: 
        ## merge all articles of a user, with high ratings, so that user is more interested to those type of articles
        if article[0]>2:
            text=text+" "+getArticleTextBasedOnArticleID(article[0])
    text=text[0:1000000-100]        
    userIDFeatures[key_]=nlp_english(text).vector

joblib.dump(userIDFeatures,os.getcwd()+'/doc2Vec_PKL/userIDFeatures.pkl')
joblib.dump(featureMatrix,os.getcwd()+'/doc2Vec_PKL/featureMatrix.pkl')
    

###find similarity with user and features matrix
    
def cos_cdist(matrix, vector):
    """
    Compute the cosine distances between each row of matrix and vector
    ** give more value to less similar so have to subtract from 1 
    """
    v = vector.reshape(1, -1)
    ans= list(scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1))
    ans=np.ones(len(ans))-ans
    return ans

if 1==1:        
    similarity={}
    for users in dict_testset_interactions_of_a_user.keys():
        try:## consider only users how have high strength on any article
            similarity[users]=cos_cdist(tfidf_matrix,userIDFeatures[users])
        except:
            pass
    joblib.dump(similarity,'user_similarity_with_acticles.pkl')    
    similarity=joblib.load('user_similarity_with_acticles.pkl')
#if 1==2:
#    similarity={}
#    for users in dict_testset_interactions_of_a_user.keys():
#        similarity[users]=cos_cdist(tfidf_matrix,userFeature[uid.index(users)])
#    joblib.dump(similarity,'user_similarity_with_acticles.pkl') 
    
### get top 10 recommendations
top10_Recommendation_of_a_user={}
for user in similarity.keys():
    top_10_idx=[]
    top_10_values=[]
    top_10_idx = np.argsort(similarity[user])[-100:]
    top_10_values = [articlesDetails[i][2] for i in top_10_idx]
    top10_Recommendation_of_a_user[user]=top_10_values
        
## now check with test set for match
matchedNumerator=0
totaldenominator=0

for user in top10_Recommendation_of_a_user.keys():
    actualActiclesUserReviewed=dict_test_set_user_article[user]
    predictedResult=top10_Recommendation_of_a_user[user]
    number_of_match=len(set(actualActiclesUserReviewed)&set(predictedResult))
    matchedNumerator=matchedNumerator+number_of_match
    totaldenominator=totaldenominator+len(set(actualActiclesUserReviewed))
#    print('number_of_match for user:',  user,' is :>',number_of_match,"number of actual articles",len(actualActiclesUserReviewed))
print(" Document vector Recall",matchedNumerator/totaldenominator)