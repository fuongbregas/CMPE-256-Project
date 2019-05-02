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

conf = SparkConf().set("spark.default.parallelism", "300")
sc = SparkContext(conf = conf)
spark = SparkSession.builder.appName("gmALS").getOrCreate()

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

print('done with drop bookId userId timestamp from allDataDF')

trainDF = allDataDF.filter("time BETWEEN 0 AND 20256439").cache() # 90% of 22507155
valDF = allDataDF.filter("time BETWEEN 20256440 AND 22507154") # 10% of 22507155

print('done with train validation split')


# filter to just trainDF for globalMean
globalMean = trainDF.groupBy().avg("rating").take(1)[0][0]

print('done with globalMean calculation')

trainDF = trainDF.withColumn("globalMean", lit(globalMean)).cache()

#w = Window.partitionBy('userIdNum')
#q = Window.partitionBy('bookIdNum')

print("done with trainDF add column globalMean create and cache")


valDF = valDF.withColumn("globalMean", lit(globalMean)).cache()#.show()

print('done with add globalMean to valDF')



# add residuals column to trainDF
trainDF = trainDF.withColumn('residuals', col('rating') - col('globalMean')).cache()

# define and fit ALS model to residuals of trainDF
#als = ALS(rank = 20, regParam = 0.5, numUserBlocks=100, numItemBlocks=100, userCol = "userIdNum", itemCol = "bookIdNum", ratingCol = "residuals", coldStartStrategy = "drop")
als = ALS(rank = 20, regParam = 0.5, userCol = "userIdNum", itemCol = "bookIdNum", ratingCol = "residuals", coldStartStrategy = "drop")

print("done with als define")

model = als.fit(trainDF)

print("done with als fit")

predictions = model.transform(valDF).cache()

print("done with predictions cache")

predictions = predictions.withColumn('gMALS', col('prediction') + col('globalMean')).cache()

print('done with adding global mean column to predictions')

# calculate RMSE for gMALS model
evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "gMALS")
rmseGMALS = evaluator.evaluate(predictions)
print("Root-mean-square error globalMeanOnlyAndALS = " + str(rmseGMALS))

print('done the end')




