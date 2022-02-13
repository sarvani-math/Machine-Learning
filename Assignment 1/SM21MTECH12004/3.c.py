#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import seed
from random import randrange
from csv import reader
import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv('wine-dataset.csv')
df


# In[3]:


data = df.values
data


# In[4]:


a = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
b = df[['quality']]


# In[5]:


def csv_load(file):
	file = open(file, "rt")
	lines = reader(file)
	data = list(lines)
	return data


# In[6]:


file = 'wine-dataset.csv'
data = csv_load(file)
data


# In[7]:


def cv_splits(data, nfolds):
	data_split = list()
	data_copy = list(data)
	fold_size = int(len(data) / nfolds)
	for i in range(nfolds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(data_copy))
			fold.append(data_copy.pop(index))
		data_split.append(fold)
	return data_split


# In[8]:


s= cv_splits(data, 10)
s


# In[9]:


def calculate_accuracy(act, pred):
	k = 0
	for i in range(len(act)):
		if act[i] == pred[i]:
			k += 1
	return k / float(len(act)) * 100.0


# In[10]:


def evaluate_algo(data, algo, nfolds, *args):
	folds = cv_splits(data, nfolds)
	results = list()
	for fold in folds:
		train_df = list(folds)
		train_df.remove(fold)
		train_df = sum(train_df, [])
		test_df = list()
		for row in fold:
			row_copy = list(row)
			test_df.append(row_copy)
			row_copy[-1] = None
		pred = algo(train_df, test_df, *args)
		act = [row[-1] for row in fold]
		accuracy = calculate_accuracy(act, pred)
		results.append(accuracy)
	return results


# In[11]:


def splitting_test(index, value, data):
	left, right = list(), list()
	for row in data:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right


# In[12]:


def Gini_Index(groups, classes):
	n = float(sum([len(group) for group in groups]))
	gini = 0.0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue
		score = 0.0
		for class_value in classes:
			p = [row[-1] for row in group].count(class_value) / size
			score += p * p
		gini += (1.0 - score) * (size / n)
	return gini


# In[13]:


def get_potential_splits(data):
	class_v = list(set(row[-1] for row in data))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(data[0])-1):
		for row in data:
			groups = splitting_test(index, row[index], data)
			gini = Gini_Index(groups, class_v)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}


# In[14]:


def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)


# In[15]:


def split_node(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_potential_splits(left)
		split_node(node['left'], max_depth, min_size, depth+1)
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_potential_splits(right)
		split_node(node['right'], max_depth, min_size, depth+1)


# In[16]:


def build_tree(train, max_depth, min_size):
	root = get_potential_splits(train)
	split_node(root, max_depth, min_size, 1)
	return root


# In[17]:


def prediction(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return prediction(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return prediction(node['right'], row)
		else:
			return node['right']


# In[18]:


def Decision_Tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		pred = prediction(tree, row)
		predictions.append(pred)
	return(predictions)


# In[ ]:


seed(1)

nfolds = 10
max_depth = 8
min_size = 10
results = evaluate_algo(data, Decision_Tree, nfolds, max_depth, min_size)
print('Results: %s' % results)
print('Mean Accuracy: %.2f%%' % (sum(results)/float(len(results))))


# In[ ]:





# In[ ]:





# In[ ]:




