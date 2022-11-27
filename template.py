#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
import numpy as np

np.random.seed(seed=0)


def load_dataset(dataset_path):
	#To-Do: Implement this function
	df = pd.read_csv(dataset_path)

	return df

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	n_feats = len(dataset_df)
	n_class0 = len(dataset_df.loc[dataset_df['target'] == 0])
	n_class1 = len(dataset_df.loc[dataset_df['target'] == 1])
	
	return n_feats, n_class0, n_class1

from sklearn.model_selection import train_test_split
def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	random_state = np.random.random(size=3)
	train_test_split(dataset_df, testset_size, random_state)

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)