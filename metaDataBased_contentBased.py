# -*- coding: utf-8 -*-
##------------------------------------------------------------------------

#USER AND ARTICLE META BASED REGRESSION 

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



'''

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

joblib.dump(item_ids,'item_ids.pkl')

article_language = articles_df['lang'].tolist()

##analyse for title  1 time 2 time 3 times 
tfidf_matrix = vectorizer.fit_transform(articles_df['url'] + " " +articles_df['title'] + " " +articles_df['url'] + " " +articles_df['title']  + " " +articles_df['text'])
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix=tfidf_matrix.todense()
      
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
    
####-----------------------calculate the english vs portugese acticles reviewed by the user
dict_user_language_preference={}
total_portugese_count=0
total_eng_count=0
for user in dict_user_details_based_user_id.keys():
    countENG=0
    countPOR=0
    for interactions in dict_trainset_interactions_of_a_user[user]:
        if type(interactions)=='list':                
            for interaction in interactions:
                ## get article id
                acticleID=interaction[0]
                print("acticleID",acticleID)
                ##get  languag eof that article
                try:
                    language=dict_article_details_based_on_acticle_id[acticleID][12]
                except:
                    pass
                if language=='en':
                    total_eng_count+=1
                    countENG+=1
                elif language=='pt':                
                    total_portugese_count+=1
                    countPOR+=1   
        else:
            ## get article id
            acticleID=interactions[0]
            print("acticleID",acticleID)
            ##get  languag eof that article
            try:
                
                language=dict_article_details_based_on_acticle_id[acticleID][12]
            except:
                pass
            
            if language=='en':
                total_eng_count+=1
                countENG+=1
            elif language=='pt':                
                total_portugese_count+=1
                countPOR+=1
                        

    dict_user_language_preference[user]=countENG/(countENG+countPOR)    

     
    
def getLanguage_of_an_article(articleID):
    return dict_article_details_based_on_acticle_id[acticleID][12]

def get_languagePreference_of_a_user(uid):
    return dict_user_language_preference[uid]
    
    
### build user matric from the articles he has interacted
articleIDList= list(articlesDetails[:,2])
userIDFeatures={}
#userFeature=[]
#uid=[]

for key_ in dict_trainset_interactions_of_a_user.keys():
#    print("--------  working for user",key_,"")
    tempFeature=[]
    deno=0
    denominatorNormalization=0
    articleTouples=dict_trainset_interactions_of_a_user[key_]
    for acticle in articleTouples:           
        if len(tempFeature)==0:
            try:
                tempFeature=np.array(tfidf_matrix[articleIDList.index(acticle[0])])*acticle[1]
                denominatorNormalization+=acticle[1]
                deno+=1
            except:
#                print("Error for calculation of ",acticle[0])
                pass
        else:
            try:
                temp=np.array(tfidf_matrix[articleIDList.index(acticle[0])])*acticle[1]
                tempFeature=np.vstack((tempFeature,temp))
                denominatorNormalization+=acticle[1]
                deno+=1
            except:
#                print("Error for calculation of ",acticle[0])
                pass
    tempFeature=tempFeature.sum(axis=0)/denominatorNormalization  
    
    userIDFeatures[key_]=tempFeature

#####----------------------meta data collection -----------------------------
    
    
interactions_from_selected_users_df=interactions_from_selected_users_df.fillna(0)
alldata=interactions_from_selected_users_df.astype(str).values

allCountry=list(set(interactions_from_selected_users_df['userCountry'].astype(str)))##23
allRegions=list(set(interactions_from_selected_users_df['userRegion'].astype(str)))##70
allTimestamp=list(set(interactions_from_selected_users_df['timestamp'].astype(str)))##10
uniqueStamps=[]
for i in allTimestamp:
    uniqueStamps.append(int(int(i)/1000000)%10)
uniqueStamps=list(set(uniqueStamps)    )

allUserAgent=list(set(interactions_from_selected_users_df['userAgent']))

for index in range(len(allUserAgent)):
    allUserAgent[index]=str(allUserAgent[index]).split(" ")[0]
allUserAgent=list(set(allUserAgent)    )

#########################   CREATE THE FEATURE MATRIX ###################################

## random shuffle the train data
np.random.shuffle(alldata)
## take top 1000

#alldata=alldata[0:1000]

featureMatrix=[]
label=[]
processedCount=0
totalDataset=len(alldata)

for j in range(int(totalDataset/4)):
    i=alldata[j]
    processedCount+=1
    try:
        temp=[]
        ## time stamp
        timeStampValue=int((int(i[0])/1000000)%10)
        timeStamp=[0 for i in range (10)]
        timeStamp[timeStampValue]=1
        ##articleID
        artickleValue=i[2]
        articleIdFeature=list(np.array(tfidf_matrix[articleIDList.index(int(artickleValue))]).ravel())
        ## user id
        userIDvalue=i[3]
        userIDFeature=userIDFeatures[int(userIDvalue)]
        ##userAgent
        userAgentvalue=i[5]
        userAgentvalue=userAgentvalue.split(" ")[0]
        userAgentFeature=[0 for i in range(4)]
        userAgentFeature[allUserAgent.index(userAgentvalue)]=1
        ##userREgion
        userRegion=i[6]
        userRegionFeature=[0 for i in range(70)]
        userRegionFeature[allRegions.index(userRegion)]=1    
        ##userCountry
        userCountry=i[7]
        userCountryFeature=[0 for i in range(23)]
        userCountryFeature[allCountry.index(userCountry)]=1  
        ## classLebel
        label=[int(float(i[8]))]
        ##EvenStrength              
        temp=timeStamp+list(articleIdFeature)+list(userIDFeature)+userAgentFeature+userRegionFeature+userCountryFeature+label
        print(400*processedCount/totalDataset," % processed ")
        if len(featureMatrix)==0:
            featureMatrix=temp
        else:
            featureMatrix=np.vstack((featureMatrix,temp))
           
    except:
         print("exception")
joblib.dump(featureMatrix,'featureMatrix_1.pkl')        
    
'''    
######################################################################################################################
##                          classification with cross svalidation
######################################################################################################################

def sklearnRMSE(p,a):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(p,a)


def NMAE(p,a):
    nmae=0
    for i in range(len(p)):
        nmae+=abs(p[i]-a[i])
    nmae=nmae/(len(p)  *2  )
    return nmae

from sklearn.linear_model import LogisticRegression  
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer




def classifiers(trainingRatingmatrix,testingRatingMatrix,trainLabel,testLabel):
        ### logistic regression
        clf = LogisticRegression(solver='lbfgs',multi_class='multinomial',class_weight='balanced').fit(trainingRatingmatrix,trainLabel)
        acc=clf.score(testingRatingMatrix, testLabel)
        prediction=clf.predict(testingRatingMatrix)        
        print("LOgictic Regression Accuracy",acc,"RMSE",sklearnRMSE(prediction,testLabel),"NMAE:",NMAE(prediction,testLabel))
        ## LDA---------------------------------------------------------------
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf2 = LinearDiscriminantAnalysis(solver='svd')
        clf2.fit(trainingRatingmatrix, trainLabel)
        acc=clf2.score(testingRatingMatrix, testLabel)
        LDA_prediction=clf2.predict(testingRatingMatrix)        
        print("LDA::Accuracy",acc,"RMSE",sklearnRMSE(LDA_prediction,testLabel),"NMAE:",NMAE(LDA_prediction,testLabel))
        ##PCA---------------------------------------------------------------
        pca = PCA(n_components=1000)
        pca.fit(trainingRatingmatrix)
        PCA_train=pca.transform(trainingRatingmatrix)
        PCA_test=pca.transform(testingRatingMatrix)
        clf=LogisticRegression(solver='lbfgs',multi_class='multinomial')
        clf.fit(PCA_train,trainLabel)
        acc=clf.score(PCA_test,testLabel)
        prediction=clf.predict(PCA_test)  
        print("PCA: Accuracy",acc,"RMSE",sklearnRMSE(prediction,testLabel),"NMAE:",NMAE(prediction,testLabel))
        
        ## MLP classifier---------------------------------------------------------------
        clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(1000, 100), random_state=1)
        clf_mlp.fit(trainingRatingmatrix,trainLabel)
        acc=clf_mlp.score(testingRatingMatrix,testLabel)
        prediction=clf_mlp.predict(testingRatingMatrix)
        print("MLP: Accuracy",acc,"RMSE",sklearnRMSE(prediction,testLabel),"NMAE:",NMAE(prediction,testLabel))
        
        ## ELM---------------------------------------------------------------
        nh = 100
        srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)
        name = ["rbf(0.1))"]    
        classifiers = [GenELMClassifier(hidden_layer=srhl_rbf)]
        for classifier, clf in zip(name, classifiers):
            clf.fit(trainingRatingmatrix,trainLabel)
            prediction=clf.predict(testingRatingMatrix)
            score = clf.score(testingRatingMatrix, testLabel)
        
            print('ELM Model %s Accuracy: %s' % (classifier, score),"RMSE",sklearnRMSE(prediction,testLabel),"NMAE",NMAE(prediction,testLabel))
        ########
        print("===========================================================================")
             
#### combine the feature matrix
        
featureMatrix_1=joblib.load('featureMatrix_1.pkl')
featureMatrix_2=joblib.load('featureMatrix_2.pkl')
featureMatrix_3=joblib.load('featureMatrix_3.pkl')
featureMatrix_4=joblib.load('featureMatrix_4.pkl')
 
featureMatrix=np.vstack((featureMatrix_1,featureMatrix_2))
featureMatrix=np.vstack((featureMatrix,featureMatrix_3))
featureMatrix=np.vstack((featureMatrix,featureMatrix_4))    
    
    
    
features=featureMatrix[:,0:len(featureMatrix[0])-1]
labels=featureMatrix[:,len(featureMatrix)-1]

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(features):
    trainingRatingmatrix=features[train_index]
    testingRatingMatrix=features[test_index]
    print(len(testingRatingMatrix[0]))
    trainLabel=[labels[i] for i in train_index]
    testLabel=[labels[i] for i in test_index]
    classifiers(trainingRatingmatrix,testingRatingMatrix,trainLabel,testLabel)




    
    


































