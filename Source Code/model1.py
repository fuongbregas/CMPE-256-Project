import re
import sys
import math
from pyspark import sql
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import StringIndexer
from pyspark import StorageLevel
from time import time

print("done with imports")


conf = SparkConf().set("spark.default.parallelism", "300")
sc = SparkContext(conf = conf)
spark = SparkSession.builder.appName("vanillaFinal").getOrCreate()


print("done with Spark configuration")


allData = spark.read.text("ratings.csv").rdd
#allData = spark.read.text("ratings10000.csv").rdd

print("done with reading in allData")


allDataRDD = allData.map(lambda row: row.value.split(","))\
                        .map(lambda row: (str(row[0]), str(row[1]), float(row[2]), int(row[3])))\
                        .sortBy(lambda a : a[3])\
                        .zipWithIndex()\
                        .map(lambda a: (a[0][0], a[0][1], a[0][2], a[0][3], a[1]))\
                        .map(lambda a: Row(userId = str(a[0]), bookId = str(a[1]), rating = float(a[2]), timestamp = int(a[3]), time = int(a[4])))


print("done with import and parse allDataRDD")


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


#trainDataDF = allDataDF.filter("time BETWEEN 0 AND 18005724").cache() # 80% of 22507155
#testDataDF = allDataDF.filter("time BETWEEN 18005725 AND 20256439").cache() # 10% of 22507155
trainDataDF = allDataDF.filter("time BETWEEN 0 AND 20256439").cache() # 90% of 22507155
valDataDF = allDataDF.filter("time BETWEEN 20256440 AND 22507154") # 10% of 22507155

#trainDataDF = allDataDF.filter("time BETWEEN 0 AND 4999").cache()
#testDataDF = allDataDF.filter("time BETWEEN 5000 AND 9999").cache()



print("done with create and cache train and test")


als = ALS(rank = 20, regParam = 0.5, numUserBlocks=100, numItemBlocks=100, userCol = "userIdNum", itemCol = "bookIdNum", ratingCol = "rating", coldStartStrategy = "drop")


print("done with als define")


model = als.fit(trainDataDF)


print("done with als fit")


predictions = model.transform(valDataDF).cache()


print("done with predictions cache")


#predictions.write.format("csv").save(str(time()))


#print("done with predictions csv save")


evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")


print("done with evaluator define")


rmse = evaluator.evaluate(predictions)

print("Root-mean-square error = " + str(rmse))
print("Regularization parameter =" + str(als.getRegParam()))
print("Rank = " + str(als.getRank()))

print("done with rmse")

predictions.show()

allDataDF.count()
trainDataDF.count()
valDataDF.count()

print("done the end")
