import sys, os
from re import sub
from math import ceil, floor
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Window, functions

def CleanLetters(x):
	cleanLetters = x.lower()						# makes all letters lower-case
	cleanLetters = sub("[^a-z]", "", cleanLetters)				# removes all non-letter characters
	return cleanLetters							# returns clean letters
	
def CleanWords(x):
	punc='!?.,;:[]{}"()-'							# defines punctuation
	cleanWords = x.lower()							# makes all letters lower-case
	for ch in punc:
		cleanWords = cleanWords.replace(ch, ' ')			# replaces all defined punctuation marks with a space character
	cleanWords = sub("[\w]*[^a-z\s]+[\w]*", "", cleanWords)			# removes all words with non-letter characters
	return cleanWords							# returns clean words
	
def CountAll(x):
	allCount = x.count()							# counts total number of words
	return allCount								# returns count of all words/letters

def DataframeSetup(x, y, z):
	dataFrame = x.toDF(y)							# converts RDD to Data Frame for presentation

	if z == True:								# identifies the unit currenlty processed
		unit = "Word"
	else: unit = "Letter"
	
	window = Window.orderBy(functions.col("Frequency").desc(), functions.col(unit))	# creates a window
	
	dataFrame = dataFrame.withColumn("Rank", functions.row_number().over(window))	# uses window to keep track of the rows for ranking	
	dataFrame = dataFrame.select("Rank", unit, "Frequency")				# places columns in the right order
	dataFrame.createOrReplaceTempView("dfTable")					# creates a temporary view of the DF
	
def FileOutput(inputFile, isWords, allCount, distinctCount, threshold, commonL, commonH, rare, columns):
	sys.stdout = open("output.txt", "a")					# re-directs "prints" to a file
	if isWords == True:							# adds a output information header and displays data
		sys.stdout.truncate(0)
		unit = "word"
		print("-------------------------------------------")
		print("Output for " + str(inputFile))
		print("-------------------------------------------")
		print("Total number of " + unit + "s: ", allCount)
	else: unit = "letter"
	print("Number of distinct " + unit + "s: ", distinctCount)
	print("Popular threshold " + unit + ": ", threshold)
	print("Common threshold " + unit + " (lower): ", commonL)
	print("Common threshold " + unit + " (higher): ", commonH)
	print("Rare threshold " + unit + ": ", rare)
	print("-------------------------------------------")
	print("-------------------------------------------")
	print("Popular " + unit + "s")
	query = spark.sql("SELECT * FROM dfTable WHERE Rank between 1 and " + str(threshold))
	query.show(query.count(), False)
	print("Common " + unit + "s")
	query = spark.sql("SELECT * FROM dfTable WHERE Rank between " + str(commonL) + " and " + str(commonH))
	query.show(query.count(), False)
	print("Rare " + unit + "s")
	query = spark.sql("SELECT * FROM dfTable WHERE Rank between " + str(rare) + " and " + str(distinctCount))
	query.show(query.count(), False)
	print("-------------------------------------------")
	print("-------------------------------------------")
	sys.stdout.close()							# closes the file
	
	if isWords == False:							# when output is ready
		bucketPath = os.path.splitext(inputFile)[0]			# extracts S3 bucket path + initial file name without .txt
		os.system("aws s3 cp output.txt " + bucketPath + "_output.txt") # copies output file to the S3 bucket
		
def GetThresholds(x):
	threshold = ceil(x / 100 * 5)						# calculates threshold (popular)
	commonL = floor((x - threshold) / 2)					# calculates common word/letter lower threshold
	commonH = ceil((x + threshold) / 2)					# calculates common word/letter higher threshold
	rare = x - threshold							# calculates rare word/letter threshold
	return threshold, commonL, commonH, rare				# returns thresholds
		
def ProcessLetters(initial):
	isWords = False								# indicates that letters are being processed
	
	cleanLetters = initial.map(CleanLetters)				# creates RDD with clean letters
	letters = cleanLetters.flatMap(lambda line : [c for c in line])		# creates RDD with split letters
	letters = letters.filter(lambda x : x != "\t")				# removes tab character from RDD
	
	distinctLetterCount, sortedLetterCounted = SortKeyValue(letters)	# records number of distinct letters and RDD with letter+frequency
	threshold, commonL, commonH, rare = GetThresholds(distinctLetterCount)	# records thresholds
	
	columns = ["Frequency","Letter"]					# specifies column names
	DataframeSetup(sortedLetterCounted, columns, isWords)			# calls Data Frame setup
	FileOutput(inputFile, isWords, 0, distinctLetterCount, threshold, commonL, commonH, rare, columns) # calls output to the file

def ProcessWords(initial):
	isWords = True								# indicates that letters are being processed
	
	cleanWords = initial.map(CleanWords)					# calls cleaning function
	words = cleanWords.flatMap(lambda line: line.split(' ')).filter(bool)	# splits words and gets rid of 'empty' words
	
	allWordCount = CountAll(words)						# counts all words
	distinctWordCount, sortedWordCounted = SortKeyValue(words)		# records number of distinct words and RDD with word+frequency
	threshold, commonL, commonH, rare = GetThresholds(distinctWordCount)	# records thresholds
	
	columns = ["Frequency","Word"]						# specifies column names
	DataframeSetup(sortedWordCounted, columns, isWords)			# calls Data Frame setup
	FileOutput(inputFile, isWords, allWordCount, distinctWordCount, threshold, commonL, commonH, rare, columns) # calls output to the file

def SortKeyValue(x):
	keyValue = x.map(lambda unit : (unit, 1))				# pairs words/letters and 1
	eachCount = keyValue.reduceByKey(lambda a,b : a + b)			# counts occurance of each word/letter

	distinctCount = eachCount.count()					# count distinct words/letters

	valueKey = eachCount.map(lambda x : (x[1], x[0]))			# swaps word/letter and number of occurance places
	sortedCounted = valueKey.sortBy(lambda x : (-x[0], x[1]), False)	# orders pairs in ascending order

	return distinctCount, sortedCounted					# returns count of distinct units and RDD with unit+frequency

if __name__ == '__main__':

	session = SparkSession.builder.appName("CountingApp").getOrCreate()	# creates a Spark Session
	sc = session.sparkContext						# creates Spark Context
	sc.setLogLevel("ERROR")							# filters logs to record only errors
	
	spark = SparkSession(sc)						

	inputFile = sys.argv[1]							# gets initial file name
	initial = sc.textFile(inputFile)					# makes initial RDD
	
	# Words part
	ProcessWords(initial)
	# Letters part
	ProcessLetters(initial)
	
	session.stop()								# stops the Spark session
