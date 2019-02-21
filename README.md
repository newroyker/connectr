# Setup:
* custom:custom-local__5.2.x-rc-scala2.11__dev__head__4816590__557dc2a__dbr5.2x__d5c928f__format-2.lz4
* Client package (pip install -U databricks-connect-5.2.1.post2.tar.gz)
https://drive.google.com/open?id=1ArXc38DeLF28bKFpVGTAoResqEXYHB_6

## SparkR script with `spark.glm`
```
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

```

* Output:
```
✘ debajyoti.roy@C02XD2NHJGH5  ~/Dev/connectr  Rscript sparkr.R

Attaching package: ‘SparkR’

The following objects are masked from ‘package:stats’:

    cov, filter, lag, na.omit, predict, sd, var, window

The following objects are masked from ‘package:base’:

    as.data.frame, colnames, colnames<-, drop, endsWith, intersect,
    rank, rbind, sample, startsWith, subset, summary, transform, union

Spark package found in SPARK_HOME: /usr/local/lib/python2.7/site-packages/pyspark
Launching java with spark-submit command /usr/local/lib/python2.7/site-packages/pyspark/bin/spark-submit   sparkr-shell /var/folders/qb/tgq8qgvj3n39jctc7pzhgq440000gp/T//RtmpZvo5Dh/backend_port1b9dc853f1b
19/02/13 10:14:30 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
19/02/13 10:14:31 WARN MetricsSystem: Using default name SparkStatusTracker for source because neither spark.metrics.namespace nor spark.app.id is set.
19/02/13 10:14:38 WARN SparkServiceRPCClient: Now tracking server state for 97b9d89a-a767-4a45-8fa0-b95f4e14d289, invalidating prev state
Java ref type org.apache.spark.sql.SparkSession id 1
[Stage 1:>                                                          (0 + 1) / 1]View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
[Stage 2:>                                                          (0 + 1) / 1]View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
19/02/13 10:14:55 WARN Instrumentation: [a66b7d6c] regParam is zero, which might cause numerical instability and overfitting.
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
19/02/13 10:14:56 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
19/02/13 10:14:56 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
19/02/13 10:14:56 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
19/02/13 10:14:56 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeRefLAPACK
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
[Stage 7:>                                                          (0 + 1) / 1]View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi

Deviance Residuals:
(Note: These are approximate quantiles with relative error <= 0.01)
     Min        1Q    Median        3Q       Max
-1.65951  -0.48982  -0.11214   0.56093   1.47146

Coefficients:
              Estimate  Std. Error   t value    Pr(>|t|)
(Intercept)   0.886604    0.080965  10.95045  0.0000e+00
features_0   -0.060493    0.406325  -0.14888  8.8195e-01
features_1   -0.613922    0.281482  -2.18103  3.1502e-02
features_2    1.677997    0.701195   2.39305  1.8556e-02
features_3   -2.035859    0.475488  -4.28162  4.2331e-05

(Dispersion parameter for gaussian family taken to be 0.4809826)

    Null deviance: 70.538  on 105  degrees of freedom
Residual deviance: 48.579  on 101  degrees of freedom
AIC: 230.1

Number of Fisher Scoring iterations: 1

View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
  label                      features  prediction
1     0 <environment: 0x7faa4fd5a9b0>  0.80150483
2     0 <environment: 0x7faa4fe56eb8>  1.96245136
3     0 <environment: 0x7faa4f9418e8>  0.72310041
4     0 <environment: 0x7faa4fcfa628>  1.39934665
5     0 <environment: 0x7faa4ff9a168> -0.04107059
6     0 <environment: 0x7faa4fb9a8a8>  0.66246359
```

## sparklyr script:
```
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

```

* output:
```
debajyoti.roy@C02XD2NHJGH5  ~/Dev/connectr  Rscript sparklyr.R

Attaching package: ‘SparkR’

The following objects are masked from ‘package:stats’:

    cov, filter, lag, na.omit, predict, sd, var, window

The following objects are masked from ‘package:base’:

    as.data.frame, colnames, colnames<-, drop, endsWith, intersect,
    rank, rbind, sample, startsWith, subset, summary, transform, union

Spark package found in SPARK_HOME: /usr/local/lib/python2.7/site-packages/pyspark
Launching java with spark-submit command /usr/local/lib/python2.7/site-packages/pyspark/bin/spark-submit   sparkr-shell /var/folders/qb/tgq8qgvj3n39jctc7pzhgq440000gp/T//Rtmp7absp6/backend_port1c1d46b04c9d
19/02/13 10:17:27 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
19/02/13 10:17:28 WARN MetricsSystem: Using default name SparkStatusTracker for source because neither spark.metrics.namespace nor spark.app.id is set.
19/02/13 10:17:35 WARN SparkServiceRPCClient: Now tracking server state for 97b9d89a-a767-4a45-8fa0-b95f4e14d289, invalidating prev state
Java ref type org.apache.spark.sql.SparkSession id 1
Installing package into ‘/Users/debajyoti.roy/Downloads/spark-2.4.0-bin-hadoop2.7/R/lib’
(as ‘lib’ is unspecified)
trying URL 'http://cran.us.r-project.org/bin/macosx/el-capitan/contrib/3.5/Rcpp_1.0.0.tgz'
Content type 'application/x-gzip' length 4535632 bytes (4.3 MB)
==================================================
downloaded 4.3 MB


The downloaded binary packages are in
	/var/folders/qb/tgq8qgvj3n39jctc7pzhgq440000gp/T//Rtmp7absp6/downloaded_packages
Installing package into ‘/Users/debajyoti.roy/Downloads/spark-2.4.0-bin-hadoop2.7/R/lib’
(as ‘lib’ is unspecified)
trying URL 'http://cran.us.r-project.org/bin/macosx/el-capitan/contrib/3.5/sparklyr_0.9.4.tgz'
Content type 'application/x-gzip' length 3641765 bytes (3.5 MB)
==================================================
downloaded 3.5 MB


The downloaded binary packages are in
	/var/folders/qb/tgq8qgvj3n39jctc7pzhgq440000gp/T//Rtmp7absp6/downloaded_packages
19/02/13 10:17:45 ERROR RBackendHandler: startSparklyr on com.databricks.backend.daemon.driver.RDriverLocal failed
java.lang.ClassNotFoundException: com.databricks.backend.daemon.driver.RDriverLocal
	at java.net.URLClassLoader.findClass(URLClassLoader.java:382)
	at java.lang.ClassLoader.loadClass(ClassLoader.java:424)
	at java.lang.ClassLoader.loadClass(ClassLoader.java:357)
	at java.lang.Class.forName0(Native Method)
	at java.lang.Class.forName(Class.java:348)
	at org.apache.spark.util.Utils$.classForName(Utils.scala:256)
	at org.apache.spark.api.r.RBackendHandler.handleMethodCall(RBackendHandler.scala:143)
	at org.apache.spark.api.r.RBackendHandler.channelRead0(RBackendHandler.scala:108)
	at org.apache.spark.api.r.RBackendHandler.channelRead0(RBackendHandler.scala:40)
	at io.netty.channel.SimpleChannelInboundHandler.channelRead(SimpleChannelInboundHandler.java:105)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:362)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:348)
	at io.netty.channel.AbstractChannelHandlerContext.fireChannelRead(AbstractChannelHandlerContext.java:340)
	at io.netty.handler.timeout.IdleStateHandler.channelRead(IdleStateHandler.java:286)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:362)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:348)
	at io.netty.channel.AbstractChannelHandlerContext.fireChannelRead(AbstractChannelHandlerContext.java:340)
	at io.netty.handler.codec.MessageToMessageDecoder.channelRead(MessageToMessageDecoder.java:102)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:362)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:348)
	at io.netty.channel.AbstractChannelHandlerContext.fireChannelRead(AbstractChannelHandlerContext.java:340)
	at io.netty.handler.codec.ByteToMessageDecoder.fireChannelRead(ByteToMessageDecoder.java:310)
	at io.netty.handler.codec.ByteToMessageDecoder.channelRead(ByteToMessageDecoder.java:284)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:362)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:348)
	at io.netty.channel.AbstractChannelHandlerContext.fireChannelRead(AbstractChannelHandlerContext.java:340)
	at io.netty.channel.DefaultChannelPipeline$HeadContext.channelRead(DefaultChannelPipeline.java:1359)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:362)
	at io.netty.channel.AbstractChannelHandlerContext.invokeChannelRead(AbstractChannelHandlerContext.java:348)
	at io.netty.channel.DefaultChannelPipeline.fireChannelRead(DefaultChannelPipeline.java:935)
	at io.netty.channel.nio.AbstractNioByteChannel$NioByteUnsafe.read(AbstractNioByteChannel.java:138)
	at io.netty.channel.nio.NioEventLoop.processSelectedKey(NioEventLoop.java:645)
	at io.netty.channel.nio.NioEventLoop.processSelectedKeysOptimized(NioEventLoop.java:580)
	at io.netty.channel.nio.NioEventLoop.processSelectedKeys(NioEventLoop.java:497)
	at io.netty.channel.nio.NioEventLoop.run(NioEventLoop.java:459)
	at io.netty.util.concurrent.SingleThreadEventExecutor$5.run(SingleThreadEventExecutor.java:858)
	at io.netty.util.concurrent.DefaultThreadFactory$DefaultRunnableDecorator.run(DefaultThreadFactory.java:138)
	at java.lang.Thread.run(Thread.java:748)
Error in value[[3L]](cond) : Failed to start sparklyr backend:
Calls: spark_connect ... tryCatch -> tryCatchList -> tryCatchOne -> <Anonymous>
Execution halted
```

## SparkR script with `as.DataFrame`:
```
#!/usr/bin/env Rscript

Sys.setenv(SPARK_HOME='/usr/local/lib/python2.7/site-packages/pyspark')

.libPaths(c(file.path('/Users/debajyoti.roy/Downloads/spark-2.4.0-bin-hadoop2.7', 'R', 'lib'), .libPaths()))

library(SparkR)

sparkR.session()

head(faithful)

df <- as.DataFrame(faithful)

head(df)

```

* output:
```
debajyoti.roy@C02XD2NHJGH5  ~/Dev/connectr   master ●  Rscript sparkr2.R

Attaching package: ‘SparkR’

The following objects are masked from ‘package:stats’:

    cov, filter, lag, na.omit, predict, sd, var, window

The following objects are masked from ‘package:base’:

    as.data.frame, colnames, colnames<-, drop, endsWith, intersect,
    rank, rbind, sample, startsWith, subset, summary, transform, union

Spark package found in SPARK_HOME: /usr/local/lib/python2.7/site-packages/pyspark
Launching java with spark-submit command /usr/local/lib/python2.7/site-packages/pyspark/bin/spark-submit   sparkr-shell /var/folders/qb/tgq8qgvj3n39jctc7pzhgq440000gp/T//RtmpInTU9n/backend_port3a7355f21e56
19/02/20 10:58:52 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
19/02/20 10:58:53 WARN MetricsSystem: Using default name SparkStatusTracker for source because neither spark.metrics.namespace nor spark.app.id is set.
19/02/20 10:59:00 WARN SparkServiceRPCClient: Now tracking server state for f840e5bd-c850-4ff2-b7fb-7ebb3370e115, invalidating prev state
Java ref type org.apache.spark.sql.SparkSession id 1
  eruptions waiting
1     3.600      79
2     1.800      54
3     3.333      74
4     2.283      62
5     4.533      85
6     2.883      55
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
[Stage 24:>                                                         (0 + 1) / 1]  eruptions waiting
1     3.600      79
2     1.800      54
3     3.333      74
4     2.283      62
5     4.533      85
6     2.883      55
```

## SparkR script with `createDataFrame` and `collect`:
```
#!/usr/bin/env Rscript

Sys.setenv(SPARK_HOME='/usr/local/lib/python2.7/site-packages/pyspark')

.libPaths(c(file.path('/Users/debajyoti.roy/Downloads/spark-2.4.0-bin-hadoop2.7', 'R', 'lib'), .libPaths()))

library(SparkR)

sparkR.session()

head(iris)

df <- createDataFrame(iris)

showDF(df)

c <- collect(df)

head(c)

```

* output:
```
 debajyoti.roy@C02XD2NHJGH5  ~/Dev/connectr   master ●  Rscript sparkr3.R

Attaching package: ‘SparkR’

The following objects are masked from ‘package:stats’:

    cov, filter, lag, na.omit, predict, sd, var, window

The following objects are masked from ‘package:base’:

    as.data.frame, colnames, colnames<-, drop, endsWith, intersect,
    rank, rbind, sample, startsWith, subset, summary, transform, union

Spark package found in SPARK_HOME: /usr/local/lib/python2.7/site-packages/pyspark
Launching java with spark-submit command /usr/local/lib/python2.7/site-packages/pyspark/bin/spark-submit   sparkr-shell /var/folders/qb/tgq8qgvj3n39jctc7pzhgq440000gp/T//Rtmp7VwJKk/backend_port3b0037d9dc90
19/02/20 11:01:22 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
19/02/20 11:01:23 WARN MetricsSystem: Using default name SparkStatusTracker for source because neither spark.metrics.namespace nor spark.app.id is set.
19/02/20 11:01:30 WARN SparkServiceRPCClient: Now tracking server state for f840e5bd-c850-4ff2-b7fb-7ebb3370e115, invalidating prev state
Java ref type org.apache.spark.sql.SparkSession id 1
  Sepal.Length Sepal.Width Petal.Length Petal.Width Species
1          5.1         3.5          1.4         0.2  setosa
2          4.9         3.0          1.4         0.2  setosa
3          4.7         3.2          1.3         0.2  setosa
4          4.6         3.1          1.5         0.2  setosa
5          5.0         3.6          1.4         0.2  setosa
6          5.4         3.9          1.7         0.4  setosa
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
Warning messages:
1: In FUN(X[[i]], ...) :
  Use Sepal_Length instead of Sepal.Length  as column name
2: In FUN(X[[i]], ...) :
  Use Sepal_Width instead of Sepal.Width  as column name
3: In FUN(X[[i]], ...) :
  Use Petal_Length instead of Petal.Length  as column name
4: In FUN(X[[i]], ...) :
  Use Petal_Width instead of Petal.Width  as column name
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
[Stage 31:>                                                         (0 + 1) / 1]+------------+-----------+------------+-----------+-------+
|Sepal_Length|Sepal_Width|Petal_Length|Petal_Width|Species|
+------------+-----------+------------+-----------+-------+
|         5.1|        3.5|         1.4|        0.2| setosa|
|         4.9|        3.0|         1.4|        0.2| setosa|
|         4.7|        3.2|         1.3|        0.2| setosa|
|         4.6|        3.1|         1.5|        0.2| setosa|
|         5.0|        3.6|         1.4|        0.2| setosa|
|         5.4|        3.9|         1.7|        0.4| setosa|
|         4.6|        3.4|         1.4|        0.3| setosa|
|         5.0|        3.4|         1.5|        0.2| setosa|
|         4.4|        2.9|         1.4|        0.2| setosa|
|         4.9|        3.1|         1.5|        0.1| setosa|
|         5.4|        3.7|         1.5|        0.2| setosa|
|         4.8|        3.4|         1.6|        0.2| setosa|
|         4.8|        3.0|         1.4|        0.1| setosa|
|         4.3|        3.0|         1.1|        0.1| setosa|
|         5.8|        4.0|         1.2|        0.2| setosa|
|         5.7|        4.4|         1.5|        0.4| setosa|
|         5.4|        3.9|         1.3|        0.4| setosa|
|         5.1|        3.5|         1.4|        0.3| setosa|
|         5.7|        3.8|         1.7|        0.3| setosa|
|         5.1|        3.8|         1.5|        0.3| setosa|
+------------+-----------+------------+-----------+-------+
only showing top 20 rows
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
[Stage 32:>                                                         (0 + 1) / 1]  Sepal_Length Sepal_Width Petal_Length Petal_Width Species
1          5.1         3.5          1.4         0.2  setosa
2          4.9         3.0          1.4         0.2  setosa
3          4.7         3.2          1.3         0.2  setosa
4          4.6         3.1          1.5         0.2  setosa
5          5.0         3.6          1.4         0.2  setosa
6          5.4         3.9          1.7         0.4  setosa
```

## SparkR script with `lapply`:
```
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
```

* output:
```
 debajyoti.roy@C02XD2NHJGH5  ~/Dev/connectr   master ●  Rscript lapply.R

Attaching package: ‘SparkR’

The following objects are masked from ‘package:stats’:

    cov, filter, lag, na.omit, predict, sd, var, window

The following objects are masked from ‘package:base’:

    as.data.frame, colnames, colnames<-, drop, endsWith, intersect,
    rank, rbind, sample, startsWith, subset, summary, transform, union

Spark package found in SPARK_HOME: /usr/local/lib/python2.7/site-packages/pyspark
Launching java with spark-submit command /usr/local/lib/python2.7/site-packages/pyspark/bin/spark-submit   sparkr-shell /var/folders/qb/tgq8qgvj3n39jctc7pzhgq440000gp/T//RtmpQeeCbR/backend_port3e6324ce8879
19/02/20 11:33:18 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
19/02/20 11:33:19 WARN MetricsSystem: Using default name SparkStatusTracker for source because neither spark.metrics.namespace nor spark.app.id is set.
19/02/20 11:33:26 WARN SparkServiceRPCClient: Now tracking server state for f840e5bd-c850-4ff2-b7fb-7ebb3370e115, invalidating prev state
Java ref type org.apache.spark.sql.SparkSession id 1
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
[[1]]

Call:
glm(formula = Sepal.Length ~ Sepal.Width + Species, family = family,
    data = iris)

Deviance Residuals:
     Min        1Q    Median        3Q       Max
-1.30711  -0.25713  -0.05325   0.19542   1.41253

Coefficients:
                  Estimate Std. Error t value Pr(>|t|)
(Intercept)         2.2514     0.3698   6.089 9.57e-09 ***
Sepal.Width         0.8036     0.1063   7.557 4.19e-12 ***
Speciesversicolor   1.4587     0.1121  13.012  < 2e-16 ***
Speciesvirginica    1.9468     0.1000  19.465  < 2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for gaussian family taken to be 0.1918059)

    Null deviance: 102.168  on 149  degrees of freedom
Residual deviance:  28.004  on 146  degrees of freedom
AIC: 183.94

Number of Fisher Scoring iterations: 2


[[2]]

Call:
glm(formula = Sepal.Length ~ Sepal.Width + Species, family = family,
    data = iris)

Deviance Residuals:
     Min        1Q    Median        3Q       Max
-0.52652  -0.10966  -0.01230   0.07755   0.56101

Coefficients:
                  Estimate Std. Error z value Pr(>|z|)
(Intercept)        1.13033    0.35454   3.188 0.001432 **
Sepal.Width        0.13971    0.10119   1.381 0.167361
Speciesversicolor  0.26277    0.10901   2.410 0.015931 *
Speciesvirginica   0.33842    0.09587   3.530 0.000416 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for poisson family taken to be 1)

    Null deviance: 17.3620  on 149  degrees of freedom
Residual deviance:  4.5202  on 146  degrees of freedom
AIC: Inf

Number of Fisher Scoring iterations: 3
```

## SparkR script with `dapply`:
```
#!/usr/bin/env Rscript

Sys.setenv(SPARK_HOME='/usr/local/lib/python2.7/site-packages/pyspark')

.libPaths(c(file.path('/Users/debajyoti.roy/Downloads/spark-2.4.0-bin-hadoop2.7', 'R', 'lib'), .libPaths()))

library(SparkR)

sparkR.session()

df <- createDataFrame(iris)
df1 <- dapply(df, function(x) { x }, schema(df))
collect(df1)
```

* output:
```
 debajyoti.roy@C02XD2NHJGH5  ~/Dev/connectr   master  Rscript dapply.R

Attaching package: ‘SparkR’

The following objects are masked from ‘package:stats’:

    cov, filter, lag, na.omit, predict, sd, var, window

The following objects are masked from ‘package:base’:

    as.data.frame, colnames, colnames<-, drop, endsWith, intersect,
    rank, rbind, sample, startsWith, subset, summary, transform, union

Spark package found in SPARK_HOME: /usr/local/lib/python2.7/site-packages/pyspark
Launching java with spark-submit command /usr/local/lib/python2.7/site-packages/pyspark/bin/spark-submit   sparkr-shell /var/folders/qb/tgq8qgvj3n39jctc7pzhgq440000gp/T//RtmptAYEbq/backend_port112e22a543f7
19/02/21 15:46:40 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
19/02/21 15:46:41 WARN MetricsSystem: Using default name SparkStatusTracker for source because neither spark.metrics.namespace nor spark.app.id is set.
19/02/21 15:46:48 WARN SparkServiceRPCClient: Now tracking server state for 8c8115ff-0b9c-4db5-9cdd-1ad278185e04, invalidating prev state
Java ref type org.apache.spark.sql.SparkSession id 1
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
Warning messages:
1: In FUN(X[[i]], ...) :
  Use Sepal_Length instead of Sepal.Length  as column name
2: In FUN(X[[i]], ...) :
  Use Sepal_Width instead of Sepal.Width  as column name
3: In FUN(X[[i]], ...) :
  Use Petal_Length instead of Petal.Length  as column name
4: In FUN(X[[i]], ...) :
  Use Petal_Width instead of Petal.Width  as column name
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
[Stage 4:>                                                          (0 + 1) / 1]    Sepal_Length Sepal_Width Petal_Length Petal_Width    Species
1            5.1         3.5          1.4         0.2     setosa
2            4.9         3.0          1.4         0.2     setosa
3            4.7         3.2          1.3         0.2     setosa
4            4.6         3.1          1.5         0.2     setosa
5            5.0         3.6          1.4         0.2     setosa
6            5.4         3.9          1.7         0.4     setosa
7            4.6         3.4          1.4         0.3     setosa
8            5.0         3.4          1.5         0.2     setosa
9            4.4         2.9          1.4         0.2     setosa
10           4.9         3.1          1.5         0.1     setosa
11           5.4         3.7          1.5         0.2     setosa
12           4.8         3.4          1.6         0.2     setosa
13           4.8         3.0          1.4         0.1     setosa
14           4.3         3.0          1.1         0.1     setosa
15           5.8         4.0          1.2         0.2     setosa
16           5.7         4.4          1.5         0.4     setosa
17           5.4         3.9          1.3         0.4     setosa
18           5.1         3.5          1.4         0.3     setosa
19           5.7         3.8          1.7         0.3     setosa
20           5.1         3.8          1.5         0.3     setosa
21           5.4         3.4          1.7         0.2     setosa
22           5.1         3.7          1.5         0.4     setosa
23           4.6         3.6          1.0         0.2     setosa
24           5.1         3.3          1.7         0.5     setosa
25           4.8         3.4          1.9         0.2     setosa
26           5.0         3.0          1.6         0.2     setosa
27           5.0         3.4          1.6         0.4     setosa
28           5.2         3.5          1.5         0.2     setosa
29           5.2         3.4          1.4         0.2     setosa
30           4.7         3.2          1.6         0.2     setosa
31           4.8         3.1          1.6         0.2     setosa
32           5.4         3.4          1.5         0.4     setosa
33           5.2         4.1          1.5         0.1     setosa
34           5.5         4.2          1.4         0.2     setosa
35           4.9         3.1          1.5         0.2     setosa
36           5.0         3.2          1.2         0.2     setosa
37           5.5         3.5          1.3         0.2     setosa
38           4.9         3.6          1.4         0.1     setosa
39           4.4         3.0          1.3         0.2     setosa
40           5.1         3.4          1.5         0.2     setosa
41           5.0         3.5          1.3         0.3     setosa
42           4.5         2.3          1.3         0.3     setosa
43           4.4         3.2          1.3         0.2     setosa
44           5.0         3.5          1.6         0.6     setosa
45           5.1         3.8          1.9         0.4     setosa
46           4.8         3.0          1.4         0.3     setosa
47           5.1         3.8          1.6         0.2     setosa
48           4.6         3.2          1.4         0.2     setosa
49           5.3         3.7          1.5         0.2     setosa
50           5.0         3.3          1.4         0.2     setosa
51           7.0         3.2          4.7         1.4 versicolor
52           6.4         3.2          4.5         1.5 versicolor
53           6.9         3.1          4.9         1.5 versicolor
54           5.5         2.3          4.0         1.3 versicolor
55           6.5         2.8          4.6         1.5 versicolor
56           5.7         2.8          4.5         1.3 versicolor
57           6.3         3.3          4.7         1.6 versicolor
58           4.9         2.4          3.3         1.0 versicolor
59           6.6         2.9          4.6         1.3 versicolor
60           5.2         2.7          3.9         1.4 versicolor
61           5.0         2.0          3.5         1.0 versicolor
62           5.9         3.0          4.2         1.5 versicolor
63           6.0         2.2          4.0         1.0 versicolor
64           6.1         2.9          4.7         1.4 versicolor
65           5.6         2.9          3.6         1.3 versicolor
66           6.7         3.1          4.4         1.4 versicolor
67           5.6         3.0          4.5         1.5 versicolor
68           5.8         2.7          4.1         1.0 versicolor
69           6.2         2.2          4.5         1.5 versicolor
70           5.6         2.5          3.9         1.1 versicolor
71           5.9         3.2          4.8         1.8 versicolor
72           6.1         2.8          4.0         1.3 versicolor
73           6.3         2.5          4.9         1.5 versicolor
74           6.1         2.8          4.7         1.2 versicolor
75           6.4         2.9          4.3         1.3 versicolor
76           6.6         3.0          4.4         1.4 versicolor
77           6.8         2.8          4.8         1.4 versicolor
78           6.7         3.0          5.0         1.7 versicolor
79           6.0         2.9          4.5         1.5 versicolor
80           5.7         2.6          3.5         1.0 versicolor
81           5.5         2.4          3.8         1.1 versicolor
82           5.5         2.4          3.7         1.0 versicolor
83           5.8         2.7          3.9         1.2 versicolor
84           6.0         2.7          5.1         1.6 versicolor
85           5.4         3.0          4.5         1.5 versicolor
86           6.0         3.4          4.5         1.6 versicolor
87           6.7         3.1          4.7         1.5 versicolor
88           6.3         2.3          4.4         1.3 versicolor
89           5.6         3.0          4.1         1.3 versicolor
90           5.5         2.5          4.0         1.3 versicolor
91           5.5         2.6          4.4         1.2 versicolor
92           6.1         3.0          4.6         1.4 versicolor
93           5.8         2.6          4.0         1.2 versicolor
94           5.0         2.3          3.3         1.0 versicolor
95           5.6         2.7          4.2         1.3 versicolor
96           5.7         3.0          4.2         1.2 versicolor
97           5.7         2.9          4.2         1.3 versicolor
98           6.2         2.9          4.3         1.3 versicolor
99           5.1         2.5          3.0         1.1 versicolor
100          5.7         2.8          4.1         1.3 versicolor
101          6.3         3.3          6.0         2.5  virginica
102          5.8         2.7          5.1         1.9  virginica
103          7.1         3.0          5.9         2.1  virginica
104          6.3         2.9          5.6         1.8  virginica
105          6.5         3.0          5.8         2.2  virginica
106          7.6         3.0          6.6         2.1  virginica
107          4.9         2.5          4.5         1.7  virginica
108          7.3         2.9          6.3         1.8  virginica
109          6.7         2.5          5.8         1.8  virginica
110          7.2         3.6          6.1         2.5  virginica
111          6.5         3.2          5.1         2.0  virginica
112          6.4         2.7          5.3         1.9  virginica
113          6.8         3.0          5.5         2.1  virginica
114          5.7         2.5          5.0         2.0  virginica
115          5.8         2.8          5.1         2.4  virginica
116          6.4         3.2          5.3         2.3  virginica
117          6.5         3.0          5.5         1.8  virginica
118          7.7         3.8          6.7         2.2  virginica
119          7.7         2.6          6.9         2.3  virginica
120          6.0         2.2          5.0         1.5  virginica
121          6.9         3.2          5.7         2.3  virginica
122          5.6         2.8          4.9         2.0  virginica
123          7.7         2.8          6.7         2.0  virginica
124          6.3         2.7          4.9         1.8  virginica
125          6.7         3.3          5.7         2.1  virginica
126          7.2         3.2          6.0         1.8  virginica
127          6.2         2.8          4.8         1.8  virginica
128          6.1         3.0          4.9         1.8  virginica
129          6.4         2.8          5.6         2.1  virginica
130          7.2         3.0          5.8         1.6  virginica
131          7.4         2.8          6.1         1.9  virginica
132          7.9         3.8          6.4         2.0  virginica
133          6.4         2.8          5.6         2.2  virginica
134          6.3         2.8          5.1         1.5  virginica
135          6.1         2.6          5.6         1.4  virginica
136          7.7         3.0          6.1         2.3  virginica
137          6.3         3.4          5.6         2.4  virginica
138          6.4         3.1          5.5         1.8  virginica
139          6.0         3.0          4.8         1.8  virginica
140          6.9         3.1          5.4         2.1  virginica
141          6.7         3.1          5.6         2.4  virginica
142          6.9         3.1          5.1         2.3  virginica
143          5.8         2.7          5.1         1.9  virginica
144          6.8         3.2          5.9         2.3  virginica
145          6.7         3.3          5.7         2.5  virginica
146          6.7         3.0          5.2         2.3  virginica
147          6.3         2.5          5.0         1.9  virginica
148          6.5         3.0          5.2         2.0  virginica
149          6.2         3.4          5.4         2.3  virginica
150          5.9         3.0          5.1         1.8  virginica
```
