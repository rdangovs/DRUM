#Script    2/2      ---> First is splitEnWik.py
#TThis file takes the output of the splitEnWik.py file and removes all words that have a frequency of 1.
#This is after the split script


import numpy as np

input_file = "input_words_nodigits_new.txt"

output_full = "input_full.txt"
output_kicked = "input_kicked.txt"
arr = {}

def saveArray(arrayName):
	with open(output_dict,'w') as output_dict_file:
		with open(output_dict_count,'w') as output_count_file:
			newList = []
			for key, value in arrayName.iteritems():
				newList.append((value[0],key,value[1]))
			newList.sort(key = lambda x: x[1])
			#print newList
			#print arrayName
			#print arrayName.keys()
			for i in newList:
				myl = str(len(str(i[1])))
				output_dict_file.write(str(i[0]) + ' ' + repr(str(i[1])) + '\n')
				output_count_file.write(str(i[0]) + ',' + str(i[2]) + ',' + myl + '\n')
			return

with open(output_kicked,'w') as kicked_file:
	with open(output_full,'w') as full_file:
		with open(input_file,'r') as infile:
			#So lets go through the same process. First, find the vocab (aka all integers that are present), then the vocab size, then create the counts. We will assume that the vocab is just all integers from 0 to the maximum.

			#First find the max
			maxNum = 0
			myCounts = {}
			for lin in infile:
				maxNum = max(int(lin),maxNum)
				if int(lin) in myCounts:
					myCounts[int(lin)] = 1 + myCounts[int(lin)]
				else:
					myCounts[int(lin)] = 1
			print('This file has: ' , maxNum+1, " unique words")
			print("The counts of all words are: " , myCounts)
			removeSet = []
			for item, key in myCounts.iteritems():
				if (key == 1):
					removeSet.append(item)
			print('Removing: ' , len(removeSet))
			#The remove set has all the things to remove.
			newMax = maxNum+1
		with open(input_file,'r') as infile:
			for lin in infile:
				if int(lin) in removeSet:
					full_file.write(str(newMax)+'\n')
					kicked_file.write(str(int(lin))+'\n')
				else:
					full_file.write(str(int(lin)) + "\n")
			print(np.histogram(myCounts.values(),bins=[0,1,2,3,4,5,6,7,8,9,10,100]))
			
print "DoneDone"