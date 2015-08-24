if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on.
if (! ("methods" %in% rownames(installed.packages()))) { install.packages("methods") }
if (! ("statmod" %in% rownames(installed.packages()))) { install.packages("statmod") }
if (! ("stats" %in% rownames(installed.packages()))) { install.packages("stats") }
if (! ("graphics" %in% rownames(installed.packages()))) { install.packages("graphics") }
if (! ("RCurl" %in% rownames(installed.packages()))) { install.packages("RCurl") }
if (! ("rjson" %in% rownames(installed.packages()))) { install.packages("rjson") }
if (! ("tools" %in% rownames(installed.packages()))) { install.packages("tools") }
if (! ("utils" %in% rownames(installed.packages()))) { install.packages("utils") }

# Now we download, install and initialize the H2O package for R.
install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/rel-simons/7/R")))
library(h2o)
localH2O = h2o.init()

library(h2o)
setwd('~/scripts/titanic_ml_python_h2o/')
#create localh2o instance
localh2o <- h2o.init(ip = 'localhost',
                     port=54321, 
                     max_mem_size='4g',
                     nthreads = -1)

# upload data into the local h2o instances
data_path <- "data.csv"

data.hex <- h2o.importFile(localh2o,path = data_path)

# specify the data and class label
data = c(3,5,6,7,8,9,11)
class = 2

data.hex[,2] = as.factor(data.hex[,2])

# set the parameters for data split
data.split<- h2o.splitFrame(data=data.hex,ratios = 0.6)

# Assign the data as training data and testing data
train = data.split[[1]]
test = data.split[[2]]

# build the gbm 
data.gbm = h2o.gbm(training_frame=train, 
                   x=data,
                   y=class,
                   ntrees=50,
                   max.depth=5)

# print the gbm information
data.gbm

# predict the result on testing data
pred_frame = h2o.predict(data.gbm,test)

# compute the performance on testing data
#perf = h2o.performance(pred_frame[,3],test[,2])

perf = h2o.performance(data.gbm, test)
perf

# plot auc curve
plot(perf, type = "roc", col = "blue", typ = "b")

# plot the score to get the cut-off
plot(perf, type = "cutoffs", col = "blue")

# build random forest
data.rf = h2o.randomForest(data=train,
                           x = data, 
                           y = class, 
                           importance = T, 
                           ntree=200, 
                           depth=10)

# predict the result on testing data
rf_frame = h2o.predict(data.rf, test)

# compute the performance on the testing data
perf_rf = h2o.performance(rf_frame[,3],test[,2])

# plot the auc curve
plot(perf_rf, type = "roc", col = "blue", typ = "b")

# plot the score to get the cut-off
plot(perf_rf, type = "cutoffs", col = "blue")

h2o.shutdown(localh2o)