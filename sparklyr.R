#!/usr/bin/env Rscript

Sys.setenv(SPARK_HOME='/usr/local/lib/python2.7/site-packages/pyspark')

assign("DATABRICKS_GUID", 'e9022058-976d-4432-ad52-97b9465bcfff', envir = .GlobalEnv)

.libPaths(c(file.path('/Users/debajyoti.roy/Downloads/spark-2.4.0-bin-hadoop2.7', 'R', 'lib'), .libPaths()))

library(SparkR)

sparkR.session()

install.packages("Rcpp", repos = "http://cran.us.r-project.org")

install.packages("sparklyr", repos = "http://cran.us.r-project.org")

library(sparklyr)

sc <- spark_connect(method = "databricks")

library(dplyr)

iris_tbl <- copy_to(sc, iris)

iris_tbl %>% count
