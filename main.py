import os
import sys

os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ["HADOOP_HOME"] = r"C:\\hadoop"
os.environ["PATH"] = os.environ["HADOOP_HOME"] + r"\bin;" + os.environ["PATH"]
from pyspark.sql import SparkSession

try:
    spark.stop()
except:
    pass

spark = SparkSession.builder \
    .appName("TestSpark") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

print("Spark version:", spark.version)
print("Python executable:", sys.executable)
# CHARGEMENT ET EXPLORATION 

df = spark.read.csv('dataset.csv', header=True, inferSchema=True)
df.show()
df.printSchema()

df.describe().show()

from pyspark.sql.functions import col, sum as sum

df_manquantes = df.select(
    [sum(col(column).isNull().cast('int')).alias(column) for column in df.columns]
)
df_manquantes.show()
from pyspark.sql.types import NumericType

numeric_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, NumericType)]
print(numeric_cols)

import seaborn as sns 
import matplotlib.pyplot as plt

numerical_df = df.select(numeric_cols).toPandas()
plt.subplots(figsize=(30,30))
for index, col_name in enumerate(numeric_cols):
    plt.subplot(6,2, index+1)
    sns.histplot(numerical_df[col_name], kde=True)
    plt.title(f"histogram of {col_name}")
    plt.xlabel(col_name)
plt.show()
df.groupBy("gender").agg({"CustomerId": "count"}).show()

df.groupBy("age").agg({"HasCrCard": "count"}).show()

df.groupBy("gender").agg({"age": "count"}).show()

plt.Figure(figsize=(6,4))
plt.subplots(figsize=(30,30))
for index, col_name in enumerate(numerical_df.columns):
    plt.subplot(6,2, index+1)
    sns.boxplot(data=numerical_df, y=col_name)
    plt.title(col_name)
plt.show()
df = df[df["Age"] <= 80]

# NETTOYAGE ET ENCODAGE 
df = df['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
              'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
              'EstimatedSalary', 'Exited']
df.show(5)
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

cat_columns = ['Geography', 'Gender']
indexers = [StringIndexer(inputCol=col, outputCol=col+'_index') for col in cat_columns]

pipeline = Pipeline(stages=indexers)
df = pipeline.fit(df).transform(df)

df.show()

df = df.drop("Geography", "Gender")
df.show()
# MONGODB 

df_pandas = df.toPandas()

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["ma_base"]
collection = db["clients"]

collection.delete_many({})

records = df_pandas.to_dict(orient="records")
print(records[:3])

x = collection.insert_many(records)
# SMOTE 

data = list(collection.find({}, {"_id": 0}))

import pandas as pd
data = pd.DataFrame(data)

from imblearn.over_sampling import SMOTE

X = data[["CreditScore", "Age", "Tenure", "Balance",
          "NumOfProducts", "HasCrCard", "IsActiveMember",
          "EstimatedSalary", "Geography_index", "Gender_index"]]
y = data["Exited"]

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

data = pd.concat([X_res, y_res], axis=1)
print(data["Exited"].value_counts().to_dict())
# NORMALISATION 
data = spark.createDataFrame(data)

from pyspark.ml.feature import VectorAssembler, StandardScaler

feature_cols = ["CreditScore", "Age", "Tenure", "Balance",
                "NumOfProducts", "HasCrCard", "IsActiveMember",
                "EstimatedSalary", "Geography_index", "Gender_index"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(data)

data.take(5)
data.show()
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scaler_model = scaler.fit(data)
data = scaler_model.transform(data)
scaler_path = "C:/Users/abirm/Projects/BankChurnPredict/models/scaler"
scaler_model.write().overwrite().save(scaler_path)
# SÉPARATION DES DONNÉES 
train_df, test_df = data.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_df.count()}, Test: {test_df.count()}")
# ENTRAÎNEMENT 
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="Exited", seed=42)
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol="Exited", metricName="areaUnderROC")

cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3,
    seed=42,
    parallelism=1
)

cv_model = cv.fit(train_df)
# ÉVALUATION 
predictions = cv_model.transform(test_df)
predictions.select("Exited", "prediction", "probability").show(20)
auc_roc = evaluator.evaluate(predictions)
print(f"AUC-ROC: {auc_roc:.4f}")
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

multi_evaluator = MulticlassClassificationEvaluator(labelCol="Exited", predictionCol="prediction")

accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
# MATRICE DE CONFUSION 
predictions_pd = predictions.select("Exited", "prediction").toPandas()


from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns 
cm = confusion_matrix(predictions_pd["Exited"], predictions_pd["prediction"])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('matrice de confusion')
plt.ylabel('reel')
plt.xlabel('predit')
plt.show()

print("Master:", spark.sparkContext.master)
print("Partitions:", data.rdd.getNumPartitions())
model_path = "C:/Users/abirm/Projects/BankChurnPredict/models/model"
cv_model.bestModel.write().overwrite().save(model_path)
