#!/usr/bin/env Rscript

Sys.setenv(SPARK_HOME='/usr/local/lib/python2.7/site-packages/pyspark')

.libPaths(c(file.path('/Users/debajyoti.roy/Downloads/spark-2.4.0-bin-hadoop2.7', 'R', 'lib'), .libPaths()))

library(SparkR)

sparkR.session()

families <- c("gaussian", "poisson")
train <- function(family) {
  model <- glm(Sepal.Length ~ Sepal.Width + Species, iris, family = family)
  summary(model)
}
# Return a list of model's summaries
model.summaries <- spark.lapply(families, train)

# Print the summary of each model
print(model.summaries)
