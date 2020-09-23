import numpy as np
file_hampath='C:\\Workspace\\enron1\\enron1\\ham\\0007.1999-12-14.farmer.ham.txt'
file_path1='C:/Users/Administrator/Downloads/enron1'
print(file_hampath)
print(file_path1)
'''

这里面将file_path 绑定成一个对象，利用对象里提供的方法read(),去把对象里的内容读出来
'''
with open(file_hampath,'r') as infile:
     hamsample=infile.read()
print(hamsample)
print('-------------------------------------')
file_spampath='C:\\Workspace\\enron1\\enron1\\spam\\0058.2003-12-21.GP.spam.txt'

with open(file_spampath,'r') as infile:
     spamsample=infile.read()
print(spamsample)

import glob
import os
emails,labels=[],[]
file_path='C:\\Workspace\\enron1\\enron1\\spam\\'
print('**************************************')
for filename in glob.glob(os.path.join(file_path,'*.txt')):
     with open(filename,'r',encoding="ISO-8859-1") as infile:
          emails.append(infile.read())

          labels.append(1)

hamemail_len=len(emails)
print('-----SPAM---------')
print(hamemail_len)
hamlabels_len=len(labels)
print(labels)
print(hamlabels_len)
print('-----SPAM---------')
#print(emails)
file_path='C:\\Workspace\\enron1\\enron1\\ham\\'
print('**************************************')
for filename in glob.glob(os.path.join(file_path, '*.txt')):
     with open(filename, 'r', encoding="ISO-8859-1") as infile:
          emails.append(infile.read())

          labels.append(0)
'''
email append一次就加一个label，就是加了一个sample
'''
print('-----HAM---------')
hamemail_len=len(emails)
print(hamemail_len)
hamlabels_len=len(labels)
print(hamlabels_len)
print('-----HAM---------')

from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
def is_letter_only(word):
     return word.isalpha()
all_names=set(names.words())
lemmatizer=WordNetLemmatizer()

def clean_text(docs):
     docs_cleaned=[]
     for doc in docs:
          doc=doc.lower()
          doc_cleaned=''.join(lemmatizer.lemmatize(word)
                   for word in doc.split() if is_letter_only(word)
                   and word not in all_names)
          docs_cleaned.append(doc_cleaned)
     return docs_cleaned
emails_cleaned=clean_text(emails)

#print(emails_cleaned)


from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(stop_words="english",max_features=1000,max_df=0.5,min_df=2)
docs_cv=cv.fit_transform(emails_cleaned)
print(docs_cv.shape)
#print(docs_cv[1])

terms=cv.get_feature_names()
#print(terms.size())
print(len(terms))
print(terms[277])
print()
print(terms[3])

#288 term

def get_label_index(labels):
     from collections import defaultdict
     label_index=defaultdict(list)
     for index ,label in enumerate(labels):
          label_index[label].append(index)
         #很巧妙的把这里的0,1写进index,就是说Index的序列标志了0类和1类
     return label_index
label_index=get_label_index(labels)

print(labels)
print(len(labels))
print(label_index)
print(len(label_index.values()))
print("###############")
print(label_index.values())
print("###############")
def get_prior(label_index):
     prior={label:len(index) for label,index in label_index.items()}
     print("----------------prior--------------")
     print(prior)
     print("----------------prior--------------")
     print(prior.values())
     #print(prior.values()[1])
     total_count=sum(prior.values())
     for label in prior:
          prior[label]/=float(total_count)
         #prior[label]=prior[label]/float(total_count)
     return prior
prior=get_prior(label_index)
#{1:[0,1,3],0:}
print("Prior:",prior)
print(label_index)
print("****************######******************")
print(docs_cv)
print("**************#######********************")
print(docs_cv[1,:])
print("**************######********************")
def get_likelihood(term_matrix,label_index,smoothing=0):
     likelihood={}
     for label,index in label_index.items():
          print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
          print("______________________________________________________",label)
          print(label)
          print(index)
          print(term_matrix[index,:])
          print(term_matrix[index,:].shape)
          print(term_matrix[index,:].sum(axis=0))
          print(term_matrix[index,:].sum(axis=0).shape)

          print("______________________________________________________")
          likelihood[label]=term_matrix[index,:].sum(axis=0)+smoothing
          likelihood[label]=np.asarray(likelihood[label])[0]
          print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
          print( likelihood[label])
          print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
          print("^^^^^^^^^^^^^^^^^^^^Likelihood^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
          print(likelihood[label])
          print("^^^^^^^^^^^^^^^^^^^^Likelihood^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
          total_count=likelihood[label].sum()
          likelihood[label]=likelihood[label]/float(total_count)
          print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
     return likelihood
smoothing=1


likelihood=get_likelihood(docs_cv,label_index,smoothing)
#for label=1  (1500,288)-->(1,288) for label=0 (3672, 288)-->(1,288)
print(len(likelihood[0]))
print(likelihood[0])
print("---------------------")

print(len(likelihood[0][:5]))
print(likelihood[0][:10])
print("**************************")
print(likelihood[1][:20])
print("****************The likelihood is****************")
print(likelihood)
print("****************The likelihood is****************")
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||")
print(likelihood[0])
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||")

print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||")
print(label_index)
print("###############################")
print(label_index[0])

print(label_index[1])
print(likelihood[1].shape)
print(likelihood[1].sum())
#print(likelihood[1].sum(axis=1))
print(likelihood[1].sum(axis=0))
print(docs_cv.shape)
print(docs_cv.shape[0])
print(docs_cv.shape[1])

#print(docs_cv)
#print(docs_cv.shape)

#print(docs_cv.getrow(3).data)

#docs_test=cv.fit_transform(emails_cleaned)
#print(docs_test.data)
#print(docs_test.indices)
##print(docs_test.shape)
def get_posterior(term_matrix,prior,likelihood):
     num_docs=term_matrix.shape[0]
     print("The number of num_docs is")
     print(num_docs)
     posteriors=[]
     for i in range(num_docs):
          posterior={key:np.log(prior_label) for key,prior_label in prior.items()}
          print("~~~~~~~~~~~~~~~~~~~~~~~~Eingang~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
          print("The posterior is :",posterior)
          for label,likelihood_label in likelihood.items():
               term_document_vector=term_matrix.getrow(i)
               print("++++++++++++---------------------------------------++++++++++")
               print("The term_document_vector is ",term_document_vector)

               counts=term_document_vector.data
               print("The counts is ", counts)
               indices=term_document_vector.indices
               print("The indices is ", indices)
               print("++++++++++++---------------------------------------++++++++++")
               for count,index in zip(counts,indices):
                    print("***************************************")
                    print(count)
                    print(indices)
                    posterior[label]+=np.log(likelihood_label[index])*count
                    print("Label:",label,"The posterior[",label,"]",posterior[label])
                    print("***************************************")
          print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
          min_log_posterior=min(posterior.values())
          print(min_log_posterior)
          print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
          for label in posterior:
               try:
                    posterior[label]=np.exp(posterior[label]-min_log_posterior)
               except:
                    posterior[label]=float('inf')
          print("The posterior value is :",posterior.values())
          sum_posterior=sum(posterior.values())
          print("sum_posterior is :",sum_posterior)
          for label in posterior:
               if posterior[label]==float('inf'):
                  posterior[label]=1
               else:
                  posterior[label]/=sum_posterior
          posteriors.append(posterior.copy())
     return posteriors
print(get_posterior(docs_cv,prior,likelihood))
