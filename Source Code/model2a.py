import re
import sys
import math
from pyspark import sql
from pyspark.sql.functions import monotonically_increasing_id, lit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, Window
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import StringIndexer
from pyspark import StorageLevel
from time import time
from pyspark.sql.functions import mean as _mean, stddev as _stddev, col
from pyspark.sql.functions import *

print("done with imports")

conf = SparkConf()
sc = SparkContext(conf = conf)
spark = SparkSession.builder.appName("Jupyter 190501 Bias + ALS Recommendation").getOrCreate()

print("done with Spark configuration")

# read in dataset to RDD
allData = spark.read.text("ratings.csv").rdd

print('done with read in dataset to RDD')

allDataRDD = allData.map(lambda row: row.value.split(","))\
                        .map(lambda row: (str(row[0]), str(row[1]), float(row[2]), int(row[3])))\
                        .sortBy(lambda a : a[3])\
                        .zipWithIndex()\
                        .map(lambda a: (a[0][0], a[0][1], a[0][2], a[0][3], a[1]))\
                        .map(lambda a: Row(userId = str(a[0]), bookId = str(a[1]), rating = float(a[2]), timestamp = int(a[3]), time = int(a[4])))


print('done with parse allDataRDD')

# create dataframe from RDD
allDataDF = spark.createDataFrame(allDataRDD)

print("done with create allDataDF")

# convert user and item ids into integer so ALS is happy
stringIndexer = StringIndexer(inputCol="userId", outputCol="userIdNum")
model = stringIndexer.fit(allDataDF)
allDataDF = model.transform(allDataDF)


print("done with string indexer 1 of 2")


stringIndexer = StringIndexer(inputCol="bookId", outputCol="bookIdNum")
model  = stringIndexer.fit(allDataDF)
allDataDF = model.transform(allDataDF).cache()


print("done with string indexer 2 of 2")

# get rid of now unnecessary columns
allDataDF = allDataDF.drop('bookId', 'userId', 'timestamp')
allDataDF.show()

print('done with drop bookId userId timestamp from allDataDF')

#split into train, validation and test
#trainDF = allDataDF#.filter('time BETWEEN 0 AND 9999')
#valDF = allDataDF#.filter('time BETWEEN 9000 AND 9999')

trainDF = allDataDF.filter("time BETWEEN 0 AND 20256439").cache() # 90% of 22507155
valDF = allDataDF.filter("time BETWEEN 20256440 AND 22507154") # 10% of 22507155


#print(trainDF.count())
#print(valDF.count())

print('done with train validation split')


# filter to just trainDF for globalMean
globalMean = trainDF.groupBy().avg("rating").take(1)[0][0]

print('done with globalMean calculation')


# find user mean and book mean of trainDF and add to trainDF
# add columns with globalMean, userMean and bookMean to trainDF

trainDF = trainDF.withColumn("globalMean", lit(globalMean)).cache()

w = Window.partitionBy('userIdNum')
q = Window.partitionBy('bookIdNum')

trainDF = trainDF.select('time', 'userIdNum', 'bookIdNum', 'rating', 'globalMean', avg('rating').over(w).alias('userMean'))\
                    .select('time', 'userIdNum', 'bookIdNum', 'rating', 'globalMean','userMean', avg('rating').over(q).alias('bookMean'))\
                        .cache()#.sort('userIdNum', 'bookIdNum').cache()

print("done with trainDF add columns globalMean userMean bookMean create and cache")


# add columns for user bias and book bias
trainDF = trainDF.withColumn('userBias', col('userMean') - col('globalMean'))\
                    .withColumn('bookBias', col('bookMean') - col('globalMean'))\
                        .cache()

print('done with add userBias and bookBias columns to trainDF')


# drop now unnecessary bookMean and userMean columns
# add column summing globalMean, userBias and bookBias

trainDF = trainDF.drop('userMean', 'bookMean')\
                    .withColumn('gMuBbB', col('globalMean') + col('userBias') + col('bookBias'))\
                        .cache()

print('done with drop now unnecessary bookMean and userMean columns')


# https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame
#valDF.show()
#cond = [valDF.userIdNum == trainDF.userIdNum, df.age == df3.age]

#valDF = allDataDF.filter('time BETWEEN 9000 AND 9999')

valDF = valDF.join(trainDF, valDF.userIdNum == trainDF.userIdNum)\
                .select(valDF.time, valDF.userIdNum, valDF.bookIdNum, valDF.rating, 'userBias')\
                    .join(trainDF.drop('userBias'), valDF.bookIdNum == valDF.bookIdNum)\
                        .select(valDF.time, valDF.userIdNum, valDF.bookIdNum, valDF.rating, 'userBias', 'bookBias')\
                            .dropDuplicates()\
                                .cache()#.sort('userIdNum', 'bookIdNum').show()

valDF = valDF.withColumn("globalMean", lit(globalMean)).cache()#.show()

print('done with add globalMean, userBias and bookBias to valDF')


# add gMuBbB column to valDF
valDF = valDF.withColumn('gMuBbB', col('globalMean') + col('userBias') + col('bookBias')).cache()

print('done with add gMuBbB column to valDF')

# calculate rmse for gMOnly model & gMuBbBOnly model
#evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "globalMean")
#rmseGlobalMean = evaluator.evaluate(valDF)
#print("Root-mean-square error globalMeanOnly = " + str(rmseGlobalMean))

#evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "gMuBbB")
#rmseGMuBbB = evaluator.evaluate(valDF)
#print("Root-mean-square error globalMeanAndBiasesOnly = " + str(rmseGMuBbB))


# do ALS on residuals from gMuBbB model

# add residuals column to trainDF
trainDF = trainDF.withColumn('residuals', col('rating') - col('gMuBbB')).cache()

# define and fit ALS model to residuals of trainDF
#als = ALS(rank = 20, regParam = 0.5, numUserBlocks=100, numItemBlocks=100, userCol = "userIdNum", itemCol = "bookIdNum", ratingCol = "residuals", coldStartStrategy = "drop")
als = ALS(rank = 20, regParam = 0.5, userCol = "userIdNum", itemCol = "bookIdNum", ratingCol = "residuals", coldStartStrategy = "drop")



print("done with als define")


model = als.fit(trainDF)


print("done with als fit")


predictions = model.transform(valDF).cache()


print("done with predictions cache")


predictions = predictions.withColumn('gMuBbBALS', col('prediction') + col('gMuBbB')).cache()

print('done with adding gMuBbBALS column to predictions')


# calculate RMSE for gMuBbBALS model
evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "gMuBbBALS")
rmseGMuBbBALS = evaluator.evaluate(predictions)
print("Root-mean-square error globalMeanAndBiasesAndALS = " + str(rmseGMuBbBALS))

print('done the end')




