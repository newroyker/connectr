#!/usr/bin/env Rscript

Sys.setenv(SPARK_HOME='/usr/local/lib/python2.7/site-packages/pyspark')

.libPaths(c(file.path('/Users/debajyoti.roy/Downloads/spark-2.4.0-bin-hadoop2.7', 'R', 'lib'), .libPaths()))

library(SparkR)

sparkR.session()

training <- read.df("/FileStore/tables/sample_multiclass_classification_data.txt", source = "libsvm")

# Fit a generalized linear model of family "gaussian" with spark.glm
df_list <- randomSplit(training, c(7, 3), 2)
gaussianDF <- df_list[[1]]
gaussianTestDF <- df_list[[2]]
gaussianGLM <- spark.glm(gaussianDF, label ~ features, family = "gaussian")

# Model summary
summary(gaussianGLM)

# Prediction
gaussianPredictions <- predict(gaussianGLM, gaussianTestDF)
head(gaussianPredictions)
