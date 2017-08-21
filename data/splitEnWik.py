#Script    1/2     -> Second is pruneEnWik.py
#THis file will take in enwik8_r81.smaller (or the bigger) and then go through and declare a new word any time it hits punctuation.

#Should newlines be considered as the beginnings of new words? Well....yea they should because they are punctuation. '\n' symbol.

#Update to not includ ethe symbol, make these a new word.
#All numbers there own thing now.
input_file = "input.txt"

output_dict = "input_dict_nodigits_new.txt"
output_replaced = "input_words_nodigits_new.txt"
output_dict_count = "input_dict_count_new.txt"
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

with open(output_replaced,'w') as replaced_file:
	with open(output_dict,'w') as output_dict_file:
		with open(input_file,'r') as infile:
			myguy = infile.read().lower()
			#print (myguy)
			temp = ""
			for i in xrange(0,len(myguy)):
				char = myguy[i]
				#print temp
				if (char.isalpha()):
					temp += str(char)
					pass
				else:
					if temp != '':
						a = arr.get(temp)
						if a is None:
							b = len(arr)
							arr[temp] = [b,0]
							a = arr[temp]
						arr[temp] = [a[0],a[1]+1]
						#print arr
						#print "New word: " + str(temp)
						replaced_file.write(str(arr[temp][0])+"\n")
						temp = ''
					temp = char + ''
					
					#THis is a symbol, lets make this a word 
					a = arr.get(temp)
					if a is None:
						b = len(arr)
						arr[temp] = [b,0]
						a = arr[temp]
					arr[temp] = [a[0],a[1]+1]
					replaced_file.write(str(arr[temp][0])+"\n")
					temp = ''
				#There are 103 430 933
				if i % 1000000 ==0 and i > 0:
					print "I am currently : " + str(i/10000.0) + " done!"
					print "I am ready to close"		
					#break			
					#return arr 
			#Write the arr
			saveArray(arr)
			print ("There are: " + str(len(arr)) + " distinct words!")
print "DoneDone"