#!/usr/bin/env python
# coding: utf-8

# In[122]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import random
from pprint import pprint
import math


# In[123]:


df=pd.read_csv('wine-dataset.csv')
df


# In[124]:


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df


# In[125]:


random.seed(0)
train_df, test_df = train_test_split(df, test_size=0.2)


# In[126]:


data = train_df.values
data.shape


# In[127]:


def check_purity(data):
    
    quality_column = data[:, -1]
    unique_classes = np.unique(quality_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
    


# In[128]:


quality_column = data[:, -1]
unique_classes = np.unique(quality_column)
unique_classes


# In[129]:


check_purity(train_df.values)


# In[130]:


def classify_data(data):
    quality_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(quality_column, return_counts=True)
   
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification


# In[131]:


def get_potential_splits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):        
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2
                
                potential_splits[column_index].append(potential_split)
    
    return potential_splits


# In[132]:


potential_splits = get_potential_splits(train_df.values)


# In[30]:


sns.lmplot(data=train_df, x= "alcohol", y= "pH", hue="quality", fit_reg=False, size=10, aspect=2)
plt.vlines(x=potential_splits[10], ymin=2.0, ymax=3.8)


# In[13]:





# In[133]:


def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    data_below = data[split_column_values <= split_value]
    data_above = data[split_column_values >  split_value]
    
    return data_below, data_above


# In[134]:


split_column = 10
split_value = 10.625
data_below, data_above = split_data(data, split_column, split_value)


# In[16]:


plotting_df = pd.DataFrame(data, columns=df.columns)
sns.lmplot(data=plotting_df, x= "alcohol", y= "pH", hue="quality", fit_reg= False, size=10, aspect=2)
plt.vlines(x=split_value, ymin=2.0, ymax=3.8)


# In[17]:


plotting_df = pd.DataFrame(data_above, columns=df.columns)
sns.lmplot(data=plotting_df, x= "alcohol", y= "pH", hue="quality", fit_reg= False, size=10, aspect=2)
plt.vlines(x=split_value, ymin=2.0, ymax=3.8)
plt.xlim(8,14)


# In[18]:


plotting_df = pd.DataFrame(data_below, columns=df.columns)
sns.lmplot(data=plotting_df, x= "alcohol", y= "pH", hue="quality", fit_reg= False, size=10, aspect=2)
plt.vlines(x=split_value, ymin=2.0, ymax=3.8)
plt.xlim(8,14)


# In[135]:


def calculate_entropy(data):
    
    quality_column = data[:, -1]
    _, counts = np.unique(quality_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy


# In[136]:


calculate_entropy(data_above)


# In[20]:





# In[137]:


def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy 


# In[138]:


calculate_overall_entropy(data_below, data_above)


# In[139]:


def determine_best_split(data, potential_splits):
    
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)

            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


# In[140]:


potential_splits = get_potential_splits(data)


# In[141]:


determine_best_split(data, potential_splits)


# # Decision Tree Algorithm  

# In[142]:


def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=1000):
    
   
    if counter == 0:
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:
        data = df           
    
    
  
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        
        return classification

    
    
    else:    
        counter += 1

         
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        
        feature_name = COLUMN_HEADERS[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_tree = {question: []}
        
        
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)
        
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree
    


# In[143]:


tree = decision_tree_algorithm(train_df, min_samples = 2)
pprint(tree)


# In[144]:


example = test_df.iloc[800]
example


# In[145]:


def classifySample(sample, decisionTree):
   if not isinstance(decisionTree, dict):
       return decisionTree
   question = list(decisionTree.keys())[0]
   attribute, value = question.split(" <= ")
   if sample[attribute] <= float(value):
       answer = decisionTree[question][0]
   else:
       answer = decisionTree[question][1]
   return classifySample(sample, answer)


# In[146]:


def calculate_accuracy(test_df, tree):

    test_df["classification"] = test_df.apply(classifySample, axis=1, args=(tree,))
    test_df["classification_correct"] = test_df["classification"] == test_df["quality"]
    
    accuracy = test_df["classification_correct"].mean()
    
    return accuracy


# In[147]:


calculate_accuracy(test_df, tree)


# In[ ]:





# ## 3b k-fold Cross validation
# 

# In[ ]:





# In[109]:


numFolds = 10
dataset = df.sample(frac=1)
    


# In[114]:


splits = np.array_split(dataset, numFolds)
accuracies = []
for i, test_df in enumerate(splits):
    train_df = pd.concat(splits[:i] + splits[i+1:])
    acc = accuracy(train_df, test_df)
    accuracies.append(acc)
    print(np.mean(accuracies))


# ## 3.c Post pruning & Gini-Index

# In[ ]:





# In[ ]:





# In[ ]:




