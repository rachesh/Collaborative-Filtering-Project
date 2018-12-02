import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.sparse.linalg import svds
import numpy
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class Operation:
    contentStrengthFramework = '' #used to store Pandas Framework corresponding to each content popularity
    contentStrengthRegionAndCountryWise = '' #used to store Pandas Framework corresponding to each content popularity specific to their region
    personCountryAndRegionDict = {}
    ratingsDf = ''
    contentBasedSimilarity = {}
    itemList = []
    nmfmodel = ''

    def __init__(self):
        pass

    def setItemList(self,itemList):
        self.itemList = itemList

    def setContentBasedSimilarity(self,contentBasedSimilarity):
        self.contentBasedSimilarity=contentBasedSimilarity

    def performanceMatrixFormation(self,userContentInteractionFramework):
        self.contentStrengthFramework = userContentInteractionFramework.groupby('contentId')['eventStrength'].sum().sort_values(ascending=False).reset_index()
        self.contentStrengthRegionAndCountryWise = userContentInteractionFramework.groupby(['userCountry','userRegion','contentId'])['eventStrength'].sum().sort_values(ascending=False).reset_index()

    def personDictFormation(self,userContentInteractionFramework):
        for i in range(len(userContentInteractionFramework)):
            try:
                personId = userContentInteractionFramework.loc[i]['personId']
                # print(personId)
                if personId not in self.personCountryAndRegionDict:
                    # print(personId,"first condition")
                    self.personCountryAndRegionDict[personId]={}
                    self.personCountryAndRegionDict[personId]['Country']=''
                    self.personCountryAndRegionDict[personId]['Region'] = ''

                if type(userContentInteractionFramework.loc[i]['userCountry'])!=float:
                    # print(personId,"second condition")
                    self.personCountryAndRegionDict[personId]['Country'] = userContentInteractionFramework.loc[i]['userCountry']

                if type(userContentInteractionFramework.loc[i]['userRegion'])!=float:
                    # print(personId,"third condition")
                    self.personCountryAndRegionDict[personId]['Region'] = userContentInteractionFramework.loc[i]['userRegion']
            except:
                print("error occured at i th iteration ",i)

    def matrixFactorization(self,userContentInteractionFrameworkTrain,numOfFactor):
        matrixDF = userContentInteractionFrameworkTrain.pivot(index='personId',columns='contentId',values='eventStrength').fillna(0)
        matrix = matrixDF.as_matrix()
        u,sigma,vT = svds(matrix,k=numOfFactor)
        sigma = numpy.diag(sigma)
        predictedMatrix = numpy.dot(numpy.dot(u,sigma),vT)
        self.ratingsDf = pd.DataFrame(predictedMatrix, columns=matrixDF.columns,index=list(matrixDF.index)).transpose()

    def matrixFactorizationCluster(self,userContentInteractionFrameworkTrain,numOfFactor,k=2):
        matrixDF = userContentInteractionFrameworkTrain.pivot(index='personId',columns='contentId',values='eventStrength').fillna(0)
        matrix = matrixDF.as_matrix()
        a=[]
        b=[]
        kmeans = KMeans(n_clusters=k, random_state=0).fit(matrix)
        for labels in kmeans.labels_:
            if labels == 1:
                b.append(labels)
            else:
                a.append(labels)
        u,sigma,vT = svds(matrix[a],k=numOfFactor)
        sigma = numpy.diag(sigma)
        predictedMatrixA = numpy.dot(numpy.dot(u,sigma),vT)
        u,sigma,vT = svds(matrix[b],k=numOfFactor)
        sigma = numpy.diag(sigma)
        predictedMatrixB = numpy.dot(numpy.dot(u,sigma),vT)
        predictedMatrix = numpy.zeros(shape=(matrix.shape))
        predictedMatrix[a] = predictedMatrixA
        predictedMatrix[b] = predictedMatrixB
        self.ratingsDfCluster = pd.DataFrame(predictedMatrix, columns=matrixDF.columns,index=list(matrixDF.index)).transpose()

    def matrixFactorizationNMF(self,userContentInteractionFrameworkTrain,numOfFactor):
        matrixDF = userContentInteractionFrameworkTrain.pivot(index='personId',columns='contentId',values='eventStrength').fillna(0)
        matrix = matrixDF.as_matrix()
        self.nmfmodel = NMF(n_components=numOfFactor)
        W = self.nmfmodel.fit_transform(matrix)
        self.matrixnmf = self.nmfmodel.inverse_transform(W)
        self.matrixnmf = pd.DataFrame(self.matrixnmf, columns=matrixDF.columns, index=list(matrixDF.index)).transpose()

    def recommendation(self,personId,userContentInteractionFrameworkTrain,topk,isRegionMatter=True):
        contentList = userContentInteractionFrameworkTrain[userContentInteractionFrameworkTrain['personId']==personId]['contentId'].tolist()
        recommended = []
        curk = topk
        # print(topk)
        # personId = int(personId)
        if personId in self.personCountryAndRegionDict and isRegionMatter:
            if self.personCountryAndRegionDict[personId]['Region']!='':
                contentStrengthRegionAndCountryWise = self.contentStrengthRegionAndCountryWise[-self.contentStrengthRegionAndCountryWise.contentId.isin(contentList)]
                recommendedRegionWise = contentStrengthRegionAndCountryWise[contentStrengthRegionAndCountryWise['userRegion']==self.personCountryAndRegionDict[personId]['Region']]['contentId'].head(curk).tolist()
                recommended.extend(list(set(recommendedRegionWise)))
                curk = topk - len(recommended)
                # print('first condition',len(recommended),len(list(set(recommendedRegionWise))),curk,topk)
            if curk<=0:
                return recommended
            contentList.extend(recommended)

            if self.personCountryAndRegionDict[personId]['Country']!='':
                contentStrengthRegionAndCountryWise = self.contentStrengthRegionAndCountryWise[-self.contentStrengthRegionAndCountryWise.contentId.isin(contentList)]
                recommendedCountryWise = contentStrengthRegionAndCountryWise[contentStrengthRegionAndCountryWise['userCountry']==self.personCountryAndRegionDict[personId]['Country']]['contentId'].head(curk).tolist()
                recommendedCountryWise = list(set(recommendedCountryWise))
                recommended.extend(recommendedCountryWise)
                contentList.extend(recommendedCountryWise)
                # print('second condition',len(recommended),len(list(set(recommendedCountryWise))),curk)
                curk = topk - len(recommended)
            if curk <= 0:
                return recommended
        # else:
        #     print(personId,"not in dict")
        while(curk>0):
            recommendedBasedOnPopularity = self.contentStrengthFramework[-self.contentStrengthFramework.contentId.isin(contentList)]['contentId'].head(curk).tolist()
            recommendedBasedOnPopularity = list(set(recommendedBasedOnPopularity))
            contentList.extend(recommendedBasedOnPopularity)
            recommended.extend(recommendedBasedOnPopularity)
            curk -=len(recommendedBasedOnPopularity)
            # print('third condition',len(recommended),len(list(set(recommendedBasedOnPopularity))),curk)
        return recommended

    def cfRecommendation(self,personId,userContentInteractionFrameworkTrain,topk):
        contentList = userContentInteractionFrameworkTrain[userContentInteractionFrameworkTrain['personId']==personId]['contentId'].tolist()
        recommended = []
        curk = topk
        # print(topk)
        # personId = int(personId)
        while(curk>0):
            ratings = self.ratingsDf[personId].sort_values(ascending=False).reset_index()
            recommendedBasedOnMatrix= ratings[-ratings.contentId.isin(contentList)]['contentId'].head(curk).tolist()
            recommendedBasedOnMatrix = list(set(recommendedBasedOnMatrix))
            contentList.extend(recommendedBasedOnMatrix)
            recommended.extend(recommendedBasedOnMatrix)
            curk -=len(recommendedBasedOnMatrix)
            # print('third condition',len(recommended),len(list(set(recommendedBasedOnPopularity))),curk)
        return recommended

    def cfRecommendationCluster(self,personId,userContentInteractionFrameworkTrain,topk):

        contentList = userContentInteractionFrameworkTrain[userContentInteractionFrameworkTrain['personId']==personId]['contentId'].tolist()
        recommended = []
        curk = topk
        # print(topk)
        # personId = int(personId)
        while(curk>0):
            ratings = self.ratingsDfCluster[personId].sort_values(ascending=False).reset_index()
            recommendedBasedOnMatrix= ratings[-ratings.contentId.isin(contentList)]['contentId'].head(curk).tolist()
            recommendedBasedOnMatrix = list(set(recommendedBasedOnMatrix))
            contentList.extend(recommendedBasedOnMatrix)
            recommended.extend(recommendedBasedOnMatrix)
            curk -=len(recommendedBasedOnMatrix)
            # print('third condition',len(recommended),len(list(set(recommendedBasedOnPopularity))),curk)
        return recommended

    def cfRecommendationNMF(self,personId,userContentInteractionFrameworkTrain,topk):
        contentList = userContentInteractionFrameworkTrain[userContentInteractionFrameworkTrain['personId']==personId]['contentId'].tolist()
        recommended = []
        curk = topk
        # print(topk)
        # personId = int(personId)
        while(curk>0):
            ratings = self.matrixnmf[personId].sort_values(ascending=False).reset_index()
            recommendedBasedOnMatrix= ratings[-ratings.contentId.isin(contentList)]['contentId'].head(curk).tolist()
            recommendedBasedOnMatrix = list(set(recommendedBasedOnMatrix))
            contentList.extend(recommendedBasedOnMatrix)
            recommended.extend(recommendedBasedOnMatrix)
            curk -=len(recommendedBasedOnMatrix)
            # print('third condition',len(recommended),len(list(set(recommendedBasedOnPopularity))),curk)
        return recommended

    def evaluation(self,userContentInteractionFrameworkTrain,userContentInteractionFrameworkTest,topk,isRegionMatter=True,recommendartionType = 1):
        toreturn = {}
        count = 0
        for idx, i in enumerate(list(userContentInteractionFrameworkTest.index.unique().values)):
            personId = userContentInteractionFrameworkTest.loc[i]['personId']
            if recommendartionType==1:
                predicted = self.recommendation(personId,userContentInteractionFrameworkTrain,topk,isRegionMatter)
            elif recommendartionType==2:
                predicted = self.cfRecommendationNMF(personId, userContentInteractionFrameworkTrain, topk)
            elif recommendartionType==3:
                predicted = self.cfRecommendationCluster(personId, userContentInteractionFrameworkTrain, topk)
            else:
                predicted = self.cfRecommendation(personId, userContentInteractionFrameworkTrain, topk)
            actual = userContentInteractionFrameworkTest[userContentInteractionFrameworkTest['personId']==personId]['contentId'].tolist()
            actual = list(set(actual))
            toreturn[personId]={}
            toreturn[personId]['predicted'] = predicted
            toreturn[personId]['actual'] = actual
            denom = topk
            if len(actual)<topk:
                denom = len(actual)
            numer = len(list(set(actual).intersection(predicted)))
            recall = numer/float(denom)
            toreturn[personId]['recall'] = recall
            toreturn[personId]['numerator'] = numer
            toreturn[personId]['denominator'] = denom
            # print(personId,recall)
            if personId not in self.personCountryAndRegionDict:
                count +=1
            # break
        print(count,len(list(userContentInteractionFrameworkTest.index.unique().values)))
        return toreturn

    def contentBasedRecommendation(self,personId, userContentInteractionFrameworkTrain, topk):
        contentList = userContentInteractionFrameworkTrain[userContentInteractionFrameworkTrain['personId'] == personId]['contentId'].tolist()
        recommended = []
        curk = topk
        # print(topk)
        # personId = int(personId)
        ratings = self.contentBasedSimilarity[personId]
        recommended = [x for _,x in sorted(zip(ratings,self.itemList),reverse=True)]
        # print('third condition',len(recommended),len(list(set(recommendedBasedOnPopularity))),curk)
        return recommended[:topk]

    def intersectionAmongList(self,listOfLists):
        toreturn = listOfLists[0]
        for lol in listOfLists:
            toreturn=list(set(toreturn).intersection(lol))
        return lol

    # weights = [CFbased, contentBased, Country And Region wise popularityBased, popularityBased
    def hybridModelEvaluation(self,userContentInteractionFrameworkTrain,userContentInteractionFrameworkTest,topk,weights = [5,2,3,0]):
        toreturn = {}
        count = 0
        for idx, i in enumerate(list(userContentInteractionFrameworkTest.index.unique().values)):
            personId = userContentInteractionFrameworkTest.loc[i]['personId']
            predicted3 = self.recommendation(personId, userContentInteractionFrameworkTrain, topk, True)
            predicted4 = self.recommendation(personId, userContentInteractionFrameworkTrain, topk, False)
            predicted1 = self.cfRecommendation(personId, userContentInteractionFrameworkTrain, topk)
            try:
                predicted2 = self.contentBasedRecommendation(personId, userContentInteractionFrameworkTrain, topk)
            except:
                predicted2=[]
            if len(predicted2)==0:
                predicted = self.intersectionAmongList([predicted1,predicted3,predicted4])
            else:
                predicted = self.intersectionAmongList([predicted1,predicted2,predicted3,predicted4])
            predicted1 = list(set(predicted1) - set(predicted))
            predicted2 = list(set(predicted2) - set(predicted))
            predicted3 = list(set(predicted3) - set(predicted))
            predicted4 = list(set(predicted4) - set(predicted))

            if len(predicted)<topk:
                if len(predicted2)==0:
                    predicted.extend(self.intersectionAmongList([predicted1,  predicted3]))
                else:
                    predicted.extend(self.intersectionAmongList([predicted1,  predicted3]))
                predicted1 = list(set(predicted1) - set(predicted))
                predicted2 = list(set(predicted2) - set(predicted))
                predicted3 = list(set(predicted3) - set(predicted))
                predicted4 = list(set(predicted4) - set(predicted))
                if len(predicted) < topk:
                    remain = topk-len(predicted)
                    for w,lol in zip(weights,[predicted1,predicted2,predicted3,predicted4]):
                        if remain<=0:
                            predicted = predicted[:topk]
                        else:
                            res = math.ceil(0.1*w*len(lol))
                            predicted.extend(lol[:res])
                            remain = topk-len(predicted)
                else:
                    predicted = predicted[:topk]
            else:
                predicted = predicted[:topk]

            actual = userContentInteractionFrameworkTest[userContentInteractionFrameworkTest['personId'] == personId][
                'contentId'].tolist()
            actual = list(set(actual))
            toreturn[personId] = {}
            toreturn[personId]['predicted'] = predicted
            toreturn[personId]['actual'] = actual
            denom = topk
            if len(actual) < topk:
                denom = len(actual)
            numer = len(list(set(actual).intersection(predicted)))
            recall = numer / float(denom)
            toreturn[personId]['recall'] = recall
            toreturn[personId]['numerator'] = numer
            toreturn[personId]['denominator'] = denom
            # print(personId,recall)
            if personId not in self.personCountryAndRegionDict:
                count += 1
            # break
        print(count, len(list(userContentInteractionFrameworkTest.index.unique().values)))
        return toreturn

    # weights = [CFbased, contentBased, Country And Region wise popularityBased, popularityBased
    def latestHybridModelEvaluation(self,userContentInteractionFrameworkTrain,userContentInteractionFrameworkTest,topk,weights = [5,5]):
        toreturn = {}
        count = 0
        for idx, i in enumerate(list(userContentInteractionFrameworkTest.index.unique().values)):
            personId = userContentInteractionFrameworkTest.loc[i]['personId']
            # predicted2 = self.recommendation(personId, userContentInteractionFrameworkTrain, topk*2, True)
            predicted1 = self.cfRecommendation(personId, userContentInteractionFrameworkTrain, topk*2)
            predicted2 = self.cfRecommendationNMF(personId, userContentInteractionFrameworkTrain, topk*2)
            predicted = self.intersectionAmongList([predicted1,predicted2])
            predicted1 = list(set(predicted1) - set(predicted))
            predicted2 = list(set(predicted2) - set(predicted))

            if len(predicted)<topk:
                for w,lol in zip(weights,[predicted1,predicted2]):
                    res = math.ceil(0.1*w*len(lol))
                    predicted.extend(lol[:res])
                    remain = topk-len(predicted)
                    if remain<=0:
                        predicted = predicted[:topk]
            else:
                predicted = predicted[:topk]

            actual = userContentInteractionFrameworkTest[userContentInteractionFrameworkTest['personId'] == personId][
                'contentId'].tolist()
            actual = list(set(actual))
            toreturn[personId] = {}
            toreturn[personId]['predicted'] = predicted
            toreturn[personId]['actual'] = actual
            denom = topk
            if len(actual) < topk:
                denom = len(actual)
            numer = len(list(set(actual).intersection(predicted)))
            recall = numer / float(denom)
            toreturn[personId]['recall'] = recall
            toreturn[personId]['numerator'] = numer
            toreturn[personId]['denominator'] = denom
            # print(personId,recall)
            #         print(count, len(list(userContentInteractionFrameworkTest.index.unique().values)))
        return toreturn

    def globalRecallCalc(self,recallDict,comp=0.1):
        numer = 0.0
        denom = 0.0
        recallList= []
        count = 0
        total = 0
        for key in recallDict:
            recallList.append(recallDict[key]['recall'])
            numer += recallDict[key]['numerator']
            denom += recallDict[key]['denominator']
            total+=1
            if recallDict[key]['recall']>=comp:
                count +=1
        try:
            val = numer / denom
        except:
            val=0
        return val, numpy.mean(recallList),numpy.median(recallList),count/float(total)

    def plotBar(self,xlabel,ylabel,label,scores,title):
        plt.bar(numpy.arange(len(label)), scores)
        plt.xlabel(xlabel, fontsize=5)
        plt.ylabel(ylabel, fontsize=5)
        plt.xticks(scores, label, fontsize=5, rotation=45)
        plt.title(title)
        plt.savefig('project/'+title+'.jpg')
        plt.show()

    def conversionIntoGlobalFormatDictionary(self,actualDict,predictedDict,topk):
        toreturn = {}
        for personId in actualDict:
            try:
                predicted =predictedDict[personId]
                actual = actualDict[personId]
                toreturn[personId] = {}
                toreturn[personId]['predicted'] = predictedDict[personId]
                toreturn[personId]['actual'] = actualDict[personId]
                denom = topk
                if len(actual) < topk:
                    denom = len(actual)
                numer = len(list(set(actual).intersection(predicted)))
                recall = numer / float(denom)
                toreturn[personId]['recall'] = recall
                toreturn[personId]['numerator'] = numer
                toreturn[personId]['denominator'] = denom
            except:
                pass
        return toreturn

obj = Operation()
obj.performanceMatrixFormation(users_interactions)
obj.personDictFormation(users_interactions)
obj.matrixFactorization(interactions_train_df,15)
obj.matrixFactorizationNMF(interactions_train_df,15)
obj.matrixFactorizationCluster(interactions_train_df,15)
