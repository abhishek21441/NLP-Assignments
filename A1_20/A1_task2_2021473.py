# Extrinsic Evaluation
## Anger Samples
angerProbMatrix = langModel.createProbMatrix(0,0.999,emotional_dict,3)
# Anger Dataset
anger_sentences = []
while(len(anger_sentences)<50):
    cur_sentence = str.join(" ",langModel.generate_sentence())
    if emotion_scores(cur_sentence)[3]['score']>0.8 and len(cur_sentence)>5:
        anger_sentences.append(cur_sentence)
        print(len(anger_sentences))
anger_sentences
with open("./anger_sentences.txt","w") as file:
    file.write(str.join("\n",anger_sentences))
sum = 0
for cur_sentence in anger_sentences:
    sum+=emotion_scores(cur_sentence)[3]['score']
print(sum/50)
## Sadness sentences
sadnessProbMatrix = langModel.createProbMatrix(0,0.99,emotional_dict,0)
# Anger Dataset
sadness_sentences = []
while(len(sadness_sentences)<50):
    cur_sentence = str.join(" ",langModel.generate_sentence()  )
    if emotion_scores(cur_sentence)[0]['score']>0.8 and len(cur_sentence)>5:
        sadness_sentences.append(cur_sentence)
        print(len(sadness_sentences))
sadness_sentences
with open("./sadness_sentences.txt","w") as file:
    file.write(str.join("\n",sadness_sentences))
sum = 0
for cur_sentence in sadness_sentences:
    sum+=emotion_scores(cur_sentence)[0]['score']
print(sum/50)
## Joy Sentences
joyProbMatrix = langModel.createProbMatrix(0,0.999,emotional_dict,1)
# Anger Dataset
joy_sentences = []
while(len(joy_sentences)<50):
    cur_sentence = str.join(" ",langModel.generate_sentence())
    if emotion_scores(cur_sentence)[1]['score']>0.8 and len(cur_sentence)>5:
        joy_sentences.append(cur_sentence)
        print(len(joy_sentences))
joy_sentences
with open("./joy_sentences.txt","w") as file:
    file.write(str.join("\n",joy_sentences))
sum = 0
for cur_sentence in joy_sentences:
    sum+=emotion_scores(cur_sentence)[1]['score']
print(sum/50)
## Love Sentences
loveProbMatrix = langModel.createProbMatrix(0,0.999,emotional_dict,2)
# Anger Dataset
love_sentences = []
while(len(love_sentences)<50):
    cur_sentence = str.join(" ",langModel.generate_sentence())
    if emotion_scores(cur_sentence)[2]['score']>0.8 and len(cur_sentence)>5:
        love_sentences.append(cur_sentence)
        print(len(love_sentences))
love_sentences
with open("./love_sentences.txt","w") as file:
    file.write(str.join("\n",love_sentences))
sum = 0
for cur_sentence in love_sentences:
    sum+=emotion_scores(cur_sentence)[2]['score']
print(sum/50)
## Fear Sentences
fearProbMatrix = langModel.createProbMatrix(0,0.999,emotional_dict,4)
# Anger Dataset
fear_sentences = []
while(len(fear_sentences)<50):
    cur_sentence = str.join(" ",langModel.generate_sentence())
    if emotion_scores(cur_sentence)[4]['score']>0.8 and len(cur_sentence)>5:
        fear_sentences.append(cur_sentence)
        print(len(fear_sentences))
fear_sentences
with open("./fear_sentences.txt","w") as file:
    file.write(str.join("\n",fear_sentences))
sum = 0
for cur_sentence in fear_sentences:
    sum+=emotion_scores(cur_sentence)[4]['score']
print(sum/50)
## Surprise Sentences
surpriseProbMatrix = langModel.createProbMatrix(0,0.999,emotional_dict,5)
# Anger Dataset
surprise_sentences = []
while(len(surprise_sentences)<50):
    cur_sentence = str.join(" ",langModel.generate_sentence())
    if emotion_scores(cur_sentence)[5]['score']>0.8 and len(cur_sentence)>5:
        surprise_sentences.append(cur_sentence)
        print(len(surprise_sentences))
with open("./surprise_sentences.txt","w") as file:
    file.write(str.join("\n",surprise_sentences))
surprise_sentences
sum = 0
for cur_sentence in surprise_sentences:
    sum+=emotion_scores(cur_sentence)[5]['score']

print(sum/50)
print(f"Emotion of Anger: {anger_sentences[1]} {emotion_scores(anger_sentences[1])}")
print(f"Emotion of Fear: {fear_sentences[0]} {emotion_scores(fear_sentences[0])}")
print(f"Emotion of Joy: {joy_sentences[0]} {emotion_scores(joy_sentences[0])}")
print(f"Emotion of Love: {love_sentences[0]} {emotion_scores(love_sentences[0])}")
print(f"Emotion of Sadness: {sadness_sentences[0]} {emotion_scores(sadness_sentences[0])}")
print(f"Emotion of Surprise: {surprise_sentences[0]} {emotion_scores(surprise_sentences[0])}")




#List of generated samples
Test_samples=['gen_love.txt','gen_anger.txt','gen_fear.txt','gen_sadness.txt','gen_joy.txt','gen_surprise.txt']
Test_data_X=[]  #storing all samples in one list
for i in Test_samples:
    f=open(i,'r')
    a=f.read().splitlines() #take list of sentences
    f.close()
    Test_data_X.extend(a)   #add to the compiled list

#Labels for samples
Test_labels=['label_love.txt','label_anger.txt','label_fear.txt','label_sadness.txt','label_joy.txt','label_surprise.txt']
y_test=[]

for i in Test_labels:
    f=open(i,'r')
    a=f.read().splitlines()
    f.close()
    y_test.extend(a)


f1=open('corpus.txt','r')
Training_data=f1.read().splitlines()    #list of sentences in training data
f1.close()

f2=open('labels.txt','r')
Labels=f2.read().splitlines()   #list of labels in training data
f2.close()
y_train=Labels


"""## **Vectorisation**"""
#Vectorising the training and testing data

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer=TfidfVectorizer()
X_test=tfidf_vectorizer.fit_transform(Test_data_X)

X_train=tfidf_vectorizer.transform(Training_data)



#Performing grid search and training SVC
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
import numpy as np

param_grid={'C':[0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svc_model=SVC()
grid_search=GridSearchCV(svc_model, param_grid, cv=5)
grid_search.fit(X_train.toarray(), y_train)

# Find the best parameters
best_params=grid_search.best_params_
print("Best Parameters:", best_params)


# Use best parameters for SVC
best_svc_model=SVC(**best_params)
best_svc_model.fit(X_train, y_train)

#Predictions from the trained SVC model
y_pred=best_svc_model.predict(X_test)

#Evaluate performance
from sklearn.metrics import accuracy_score, classification_report

accuracy=accuracy_score(y_test, y_pred)
classification_rep=classification_report(y_test, y_pred)

print("Accuracy:",accuracy)
print("Classification Report:\n",classification_rep)