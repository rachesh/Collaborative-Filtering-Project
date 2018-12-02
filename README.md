# Collaborative-Filtering-Project
This Project is developed by Rachesh Sharma and Subhankar Adak

doc2Vec.py:  This file contains the code for the content based recommendation based on Doc2Vec. Installing NLTK and spacy is a prerequisite to run this code. This can be run independently.

tf_idf_based.py: This is for the tf-idf based vector implementation, Installing NLTK is a prerequisite to run the code.

metaDataBased_contentBased: This is the combination of both tf-idf and metadata based. The feature set creation code is commented out. Actual feature set is very large 4.5 GB. We can provide in case needed. This requires high RAM capacity to train.

opeartion.py : This file controls the evaluation part over popularity, region wise popularity, collaborative filtering both approaches, hybrid methods and clustered colaborative filtering

Once you run above files store operation class object in variable. operation class present in operation.py file

    obj = operation()

Will contain dictionary corresponding to top 50 recommendation based on popularity of article in user region as well as in country 

    val50_true = obj.evaluation(interactions_train_df,interactions_test_df,50,True)

Will contain dictionary corresponding to top 50 recommendation based on overall popularity of article 

    val50_false = obj.evaluation(interactions_train_df,interactions_test_df,50,False)

Will contain dictionary corresponding to top 50 recommendation based on singular value decompositon approach

    val50_cf = obj.evaluation(interactions_train_df,interactions_test_df,50,recommendartionType=4)

Will contain dictionary corresponding to top 50 recommendation based on NMF approach

    val50_nmf = obj.evaluation(interactions_train_df,interactions_test_df,50,recommendartionType=2)

Will contain dictionary corresponding to top 50 recommendation based on hybrid clustering approach

    val50_cfcluster = obj.evaluation(interactions_train_df,interactions_test_df,50,recommendartionType=3)

Will contain dictionary corresponding to top 50 recommendation based on hybrid weightage model 

    val50_hybrid = obj.latestHybridModelEvaluation(interactions_train_df,interactions_test_df,50,[5,5])

Each dictionary consist of userid as key and another dictionary as value. The value dictionary consist of predicted recommendation, actual recommendation, recall percentage for topk, count of match as numerator and length of actual recommendation as denominator key.

These dictionary passed through gloabRecallMethod to get gloabl recall percentage, mean, median and percentage of user correctly recommended atleast 10% recommendation out of top k.

    obj.globalRecallCalc(val50_hybrid)
