# Setup:
* Cluster DBR 5.2, Python 3
* Client setup: https://docs.databricks.com/user-guide/dev-tools/db-connect.html#client-setup

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
debajyoti.roy@C02XD2NHJGH5  ~/Dev/connectr  Rscript sparkr2.R

Attaching package: ‘SparkR’

The following objects are masked from ‘package:stats’:

    cov, filter, lag, na.omit, predict, sd, var, window

The following objects are masked from ‘package:base’:

    as.data.frame, colnames, colnames<-, drop, endsWith, intersect,
    rank, rbind, sample, startsWith, subset, summary, transform, union

Spark package found in SPARK_HOME: /usr/local/lib/python2.7/site-packages/pyspark
Launching java with spark-submit command /usr/local/lib/python2.7/site-packages/pyspark/bin/spark-submit   sparkr-shell /var/folders/qb/tgq8qgvj3n39jctc7pzhgq440000gp/T//RtmpmAQ6Og/backend_port1b822d273c89
19/02/13 10:12:56 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
19/02/13 10:12:57 WARN MetricsSystem: Using default name SparkStatusTracker for source because neither spark.metrics.namespace nor spark.app.id is set.
19/02/13 10:13:06 WARN SparkServiceRPCClient: Now tracking server state for 97b9d89a-a767-4a45-8fa0-b95f4e14d289, invalidating prev state
Java ref type org.apache.spark.sql.SparkSession id 1
  eruptions waiting
1     3.600      79
2     1.800      54
3     3.333      74
4     2.283      62
5     4.533      85
6     2.883      55
19/02/13 10:13:08 WARN SparkServiceRPCClient: Syncing 2 files (3290 bytes) took 2265 ms
View job details at https://field-eng.cloud.databricks.com/?o=0#/setting/clusters/0211-191220-synod5/sparkUi
19/02/13 10:13:11 ERROR RBackendHandler: dfToCols on org.apache.spark.sql.api.r.SQLUtils failed
java.lang.reflect.InvocationTargetException
	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
	at java.lang.reflect.Method.invoke(Method.java:498)
	at org.apache.spark.api.r.RBackendHandler.handleMethodCall(RBackendHandler.scala:167)
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
Caused by: org.apache.spark.SparkException: RDD operation class org.apache.spark.api.r.RRDD is not currently supported by SQL service.

Please add this operation to logical.proto and ProtoSerializer.scala.

scala.MatchError: RRDD[1] at RDD at RRDD.scala:44 (of class org.apache.spark.api.r.RRDD)

	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD(ProtoSerializer.scala:1252)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$serializeRDDOrNull$1.apply(ProtoSerializer.scala:1263)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$serializeRDDOrNull$1.apply(ProtoSerializer.scala:1263)
	at scala.Option.map(Option.scala:146)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDDOrNull(ProtoSerializer.scala:1263)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD0(ProtoSerializer.scala:1336)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD(ProtoSerializer.scala:1242)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$serializeRDDOrNull$1.apply(ProtoSerializer.scala:1263)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$serializeRDDOrNull$1.apply(ProtoSerializer.scala:1263)
	at scala.Option.map(Option.scala:146)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDDOrNull(ProtoSerializer.scala:1263)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD0(ProtoSerializer.scala:1336)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD(ProtoSerializer.scala:1242)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan0(ProtoSerializer.scala:615)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan(ProtoSerializer.scala:288)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan0(ProtoSerializer.scala:640)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan(ProtoSerializer.scala:288)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan0(ProtoSerializer.scala:646)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan(ProtoSerializer.scala:288)
	at com.databricks.service.SparkServiceRPCClientStub$$anonfun$executePlan$1.apply(SparkServiceRPCClientStub.scala:212)
	at com.databricks.service.SparkServiceRPCClientStub$$anonfun$executePlan$1.apply(SparkServiceRPCClientStub.scala:211)
	at com.databricks.spark.util.Log4jUsageLogger.recordOperation(UsageLogger.scala:156)
	at com.databricks.spark.util.UsageLogging$class.recordOperation(UsageLogger.scala:281)
	at com.databricks.service.SparkServiceRPCClientStub.recordOperation(SparkServiceRPCClientStub.scala:47)
	at com.databricks.service.SparkServiceRPCClientStub.executePlan(SparkServiceRPCClientStub.scala:211)
	at com.databricks.service.RemoteServiceExec.computeResult(RemoteServiceExec.scala:241)
	at com.databricks.service.RemoteServiceExec.executeCollect(RemoteServiceExec.scala:272)
	at org.apache.spark.sql.execution.collect.Collector$.collect(Collector.scala:56)
	at org.apache.spark.sql.execution.collect.Collector$.collect(Collector.scala:75)
	at org.apache.spark.sql.execution.ResultCacheManager.getOrComputeResult(ResultCacheManager.scala:46)
	at org.apache.spark.sql.execution.ResultCacheManager.getOrComputeResult(ResultCacheManager.scala:39)
	at org.apache.spark.sql.execution.SparkPlan.executeCollectResult(SparkPlan.scala:312)
	at org.apache.spark.sql.Dataset.org$apache$spark$sql$Dataset$$collectResult(Dataset.scala:2827)
	at org.apache.spark.sql.Dataset.org$apache$spark$sql$Dataset$$collectFromPlan(Dataset.scala:3439)
	at org.apache.spark.sql.Dataset$$anonfun$collect$1.apply(Dataset.scala:2794)
	at org.apache.spark.sql.Dataset$$anonfun$collect$1.apply(Dataset.scala:2794)
	at org.apache.spark.sql.Dataset$$anonfun$54.apply(Dataset.scala:3423)
	at org.apache.spark.sql.execution.SQLExecution$$anonfun$withCustomExecutionEnv$1.apply(SQLExecution.scala:99)
	at org.apache.spark.sql.execution.SQLExecution$.withSQLConfPropagated(SQLExecution.scala:228)
	at org.apache.spark.sql.execution.SQLExecution$.withCustomExecutionEnv(SQLExecution.scala:85)
	at org.apache.spark.sql.execution.SQLExecution$.withNewExecutionId(SQLExecution.scala:158)
	at org.apache.spark.sql.Dataset.org$apache$spark$sql$Dataset$$withAction(Dataset.scala:3422)
	at org.apache.spark.sql.Dataset.collect(Dataset.scala:2794)
	at org.apache.spark.sql.api.r.SQLUtils$.dfToCols(SQLUtils.scala:181)
	at org.apache.spark.sql.api.r.SQLUtils.dfToCols(SQLUtils.scala)
	... 36 more
Caused by: scala.MatchError: RRDD[1] at RDD at RRDD.scala:44 (of class org.apache.spark.api.r.RRDD)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD0(ProtoSerializer.scala:1298)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD(ProtoSerializer.scala:1242)
	... 95 more
Error in handleErrors(returnStatus, conn) :
  org.apache.spark.SparkException: RDD operation class org.apache.spark.api.r.RRDD is not currently supported by SQL service.

Please add this operation to logical.proto and ProtoSerializer.scala.

scala.MatchError: RRDD[1] at RDD at RRDD.scala:44 (of class org.apache.spark.api.r.RRDD)

	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD(ProtoSerializer.scala:1252)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$serializeRDDOrNull$1.apply(ProtoSerializer.scala:1263)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$serializeRDDOrNull$1.apply(ProtoSerializer.scala:1263)
	at scala.Option.map(Option.scala:146)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDDOrNull(ProtoSerializer.scala:1263)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD0(ProtoSerializer.scala:1336)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1
Calls: head ... .local -> callJStatic -> invokeJava -> handleErrors
Execution halted
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

#c <- collect(df)

#head(c)

```

* output:
```
✘ debajyoti.roy@C02XD2NHJGH5  ~/Dev/connectr  Rscript sparkr3.R

Attaching package: ‘SparkR’

The following objects are masked from ‘package:stats’:

    cov, filter, lag, na.omit, predict, sd, var, window

The following objects are masked from ‘package:base’:

    as.data.frame, colnames, colnames<-, drop, endsWith, intersect,
    rank, rbind, sample, startsWith, subset, summary, transform, union

Spark package found in SPARK_HOME: /usr/local/lib/python2.7/site-packages/pyspark
Launching java with spark-submit command /usr/local/lib/python2.7/site-packages/pyspark/bin/spark-submit   sparkr-shell /var/folders/qb/tgq8qgvj3n39jctc7pzhgq440000gp/T//RtmpBaUmY8/backend_portc6aa1625d72b
19/02/14 10:10:56 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
19/02/14 10:10:57 WARN MetricsSystem: Using default name SparkStatusTracker for source because neither spark.metrics.namespace nor spark.app.id is set.
19/02/14 10:11:04 WARN SparkServiceRPCClient: Now tracking server state for 66609d3e-eddd-47dd-a051-be679e9e1c6a, invalidating prev state
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
19/02/14 10:11:06 ERROR RBackendHandler: showString on 18 failed
java.lang.reflect.InvocationTargetException
	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
	at java.lang.reflect.Method.invoke(Method.java:498)
	at org.apache.spark.api.r.RBackendHandler.handleMethodCall(RBackendHandler.scala:167)
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
Caused by: org.apache.spark.SparkException: RDD operation class org.apache.spark.api.r.RRDD is not currently supported by SQL service.

Please add this operation to logical.proto and ProtoSerializer.scala.

scala.MatchError: RRDD[1] at RDD at RRDD.scala:44 (of class org.apache.spark.api.r.RRDD)

	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD(ProtoSerializer.scala:1252)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$serializeRDDOrNull$1.apply(ProtoSerializer.scala:1263)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$serializeRDDOrNull$1.apply(ProtoSerializer.scala:1263)
	at scala.Option.map(Option.scala:146)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDDOrNull(ProtoSerializer.scala:1263)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD0(ProtoSerializer.scala:1336)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD(ProtoSerializer.scala:1242)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$serializeRDDOrNull$1.apply(ProtoSerializer.scala:1263)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$serializeRDDOrNull$1.apply(ProtoSerializer.scala:1263)
	at scala.Option.map(Option.scala:146)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDDOrNull(ProtoSerializer.scala:1263)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD0(ProtoSerializer.scala:1336)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD(ProtoSerializer.scala:1242)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan0(ProtoSerializer.scala:615)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan(ProtoSerializer.scala:288)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan0(ProtoSerializer.scala:424)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan(ProtoSerializer.scala:288)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan0(ProtoSerializer.scala:640)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan(ProtoSerializer.scala:288)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan0(ProtoSerializer.scala:646)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$4.apply(ProtoSerializer.scala:289)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializePlan(ProtoSerializer.scala:288)
	at com.databricks.service.SparkServiceRPCClientStub$$anonfun$executePlan$1.apply(SparkServiceRPCClientStub.scala:212)
	at com.databricks.service.SparkServiceRPCClientStub$$anonfun$executePlan$1.apply(SparkServiceRPCClientStub.scala:211)
	at com.databricks.spark.util.Log4jUsageLogger.recordOperation(UsageLogger.scala:156)
	at com.databricks.spark.util.UsageLogging$class.recordOperation(UsageLogger.scala:281)
	at com.databricks.service.SparkServiceRPCClientStub.recordOperation(SparkServiceRPCClientStub.scala:47)
	at com.databricks.service.SparkServiceRPCClientStub.executePlan(SparkServiceRPCClientStub.scala:211)
	at com.databricks.service.RemoteServiceExec.computeResult(RemoteServiceExec.scala:241)
	at com.databricks.service.RemoteServiceExec.executeCollect(RemoteServiceExec.scala:272)
	at org.apache.spark.sql.execution.collect.Collector$.collect(Collector.scala:56)
	at org.apache.spark.sql.execution.collect.Collector$.collect(Collector.scala:75)
	at org.apache.spark.sql.execution.ResultCacheManager.getOrComputeResult(ResultCacheManager.scala:46)
	at org.apache.spark.sql.execution.ResultCacheManager.getOrComputeResult(ResultCacheManager.scala:39)
	at org.apache.spark.sql.execution.SparkPlan.executeCollectResult(SparkPlan.scala:312)
	at org.apache.spark.sql.Dataset.org$apache$spark$sql$Dataset$$collectResult(Dataset.scala:2827)
	at org.apache.spark.sql.Dataset.org$apache$spark$sql$Dataset$$collectFromPlan(Dataset.scala:3439)
	at org.apache.spark.sql.Dataset$$anonfun$head$1.apply(Dataset.scala:2556)
	at org.apache.spark.sql.Dataset$$anonfun$head$1.apply(Dataset.scala:2556)
	at org.apache.spark.sql.Dataset$$anonfun$54.apply(Dataset.scala:3423)
	at org.apache.spark.sql.execution.SQLExecution$$anonfun$withCustomExecutionEnv$1.apply(SQLExecution.scala:99)
	at org.apache.spark.sql.execution.SQLExecution$.withSQLConfPropagated(SQLExecution.scala:228)
	at org.apache.spark.sql.execution.SQLExecution$.withCustomExecutionEnv(SQLExecution.scala:85)
	at org.apache.spark.sql.execution.SQLExecution$.withNewExecutionId(SQLExecution.scala:158)
	at org.apache.spark.sql.Dataset.org$apache$spark$sql$Dataset$$withAction(Dataset.scala:3422)
	at org.apache.spark.sql.Dataset.head(Dataset.scala:2556)
	at org.apache.spark.sql.Dataset.take(Dataset.scala:2770)
	at org.apache.spark.sql.Dataset.getRows(Dataset.scala:265)
	at org.apache.spark.sql.Dataset.showString(Dataset.scala:302)
	... 36 more
Caused by: scala.MatchError: RRDD[1] at RDD at RRDD.scala:44 (of class org.apache.spark.api.r.RRDD)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD0(ProtoSerializer.scala:1298)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer.withTopLevel(ProtoSerializer.scala:276)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD(ProtoSerializer.scala:1242)
	... 101 more
Error in handleErrors(returnStatus, conn) :
  org.apache.spark.SparkException: RDD operation class org.apache.spark.api.r.RRDD is not currently supported by SQL service.

Please add this operation to logical.proto and ProtoSerializer.scala.

scala.MatchError: RRDD[1] at RDD at RRDD.scala:44 (of class org.apache.spark.api.r.RRDD)

	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD(ProtoSerializer.scala:1252)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$serializeRDDOrNull$1.apply(ProtoSerializer.scala:1263)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$serializeRDDOrNull$1.apply(ProtoSerializer.scala:1263)
	at scala.Option.map(Option.scala:146)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDDOrNull(ProtoSerializer.scala:1263)
	at org.apache.spark.sql.util.ProtoSerializer.serializeRDD0(ProtoSerializer.scala:1336)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1243)
	at org.apache.spark.sql.util.ProtoSerializer$$anonfun$5.apply(ProtoSerializer.scala:1
Calls: showDF ... .local -> callJMethod -> invokeJava -> handleErrors
Execution halted
```