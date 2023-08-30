getwd()
setwd("C:\\Users\\Alta\\OneDrive\\Desktop\\Kaggggle")
train=read.csv("train.csv")
test=read.csv("test.csv")

###i have done data inspection in excel, these are my comments 
### when i sort data by total power 

#Bigggest false flag: when you sort by total stats, almost all above 580 are legendrary, except for low dragons and megas where their non mega version is not legendrary. , only dragon with 600 stats that is legendrary is latias , informations we have about him are he is psychic and 3rd generation 
#in 3rd generation we also have slaking with 670 stats , normal, no 2nd type, not being legendrary,
#
#So it looks like we can try an overfit model or a non overfit one . we have to impute missing values also. , We could also literally write a hand algorithm for this task, but tasks wants ML so be it . 
#re observe absolute deviation
#normal pokemon with no 2nd speciality 
#normals also can be not legendrary
#imputation in test is very important . 
#Rayquaza mega rayquaza which has 780 stats , is also legendrary, probably non mega rayquaza is also legendrary
# so if non mega version is legendrary then so is mega, also the reverse is true
# so then for megas if exists check legendrary version
# else if check stats - 200 

library("rio")
library("dplyr")
library("sqldf")
library("visdat")
library("naniar")
library("caret")
library("rpart")
library("rpart.plot")
library("lares")
library("mltools")
library(data.table)

#
str(train)
is.na(train[1,7:12])

sum()
train[1,!(is.na(train[1,7:12]))]

train[1,colnames(!(is.na(train[1,7:12])))]

train[164,6+which(is.na(train[164,7:12]))]

train[164,6+which(is.na(train[164,7:12]))]
sum(train[164,6+which(!is.na(!train[164,7:12]))])
train$Total[164]
### imputing the total value train, also imputing if only one other is misisng
## #we do both in same loop for speed, also parallezible code as no relation in between row
train[1,6+which(is.na(train[1,7:12]))]
for ( i in 1:nrow(train)){
  if(is.na(train$Total[i])){
    if(!any(is.na(train[i,7:12]))){
      train[i,6]=sum(train[i,7:12])
    }
  }
  if(!is.na(train$Total[i])){
    if(sum(is.na(train[i,7:12]))==1){
      train[i,6+which(is.na(train[i,7:12]))]=train$Total[i]-sum(train[i,6+which(!is.na(!train[i,7:12]))])
    }
  }
}
train[1,5+which(is.na(train[1,7:12]))]
sum(train[1,6+which(!is.na(!train[1,7:12]))])
#train[i,which(is.na(train[1,7:12]))]
### imputing the total value test
for ( i in 1:nrow(test)){
  if(is.na(test$Total[i])){
    if(!any(is.na(test[i,7:12]))){
      test[i,6]=sum(test[i,7:12])
    }
  }
  if(!is.na(test$Total[i])){
    if(sum(is.na(test[i,7:12]))==1){
      test[i,6+which(is.na(test[i,7:12]))]=test$Total[i]-sum(test[i,6+which(!is.na(!test[i,7:12]))])
    }
  }
}
### lets impute na names with i guess none? we cant predict name lol
test$Name[is.na(test$Name )] <- "none"
train$Name[is.na(train$Name )] <- "none"
### Some dont have 2nd info, that is not na, if not na impute null
test$Type.2[test$Type.2==""] = "none"
train$Type.2[train$Type.2==""] = "none"
## Lets input unknown instead of na for tpye, its rougly same as random but not adding
## noise, it usually worrks better , its better than accidentaly putting in random dragons 
## since data is small
test$Type.1[is.na(test$Type.1 )] <- "unknown"
train$Type.1[is.na(train$Type.1 )] <- "unknown"
test$Type.2[is.na(test$Type.2 )] <- "unknown"
train$Type.2[is.na(train$Type.2 )] <- "unknown"
## We can try two imputation methods missForest, or iterated imputing methods
colMeans(train[,7:12],na.rm = T)
## take a lok at correlation prob
library(corrplot)
corvar= cor(train[,6:12],use="complete.obs")
corrplot(corvar, method = 'number')
###looks predictable, they are not all similarly correlated, for each unkown
### a formula can be used 
### imputing legendrary is not best practice
### first two columns are kinda useless lets get rid of em

### Lets for each row detect if name has mega then assign it to the useless first row 
for(i in 1:nrow(train)){
  if(grepl("Mega",train$Name[i] )){
    train[i,1]=1
  }
  else{
    train[i,1]=0
  }
}
for(i in 1:nrow(test)){
  if(grepl("Mega",test$Name[i] )){
    test[i,1]=1
  }
  else{
    test[i,1]=0
  }
}

## We are kinda getting close
## We dont need to impute total since sum gives it, also legendrary since its response
## we can impute other points
## 
library(missForest)
str(train)

train$Generation=as.factor(train$Generation)
train$X=as.factor(train$X)

test$Generation=as.factor(test$Generation)
test$X=as.factor(test$X)

df_imp_tra= missForest(train[,c(1,7:13)])
df_imp_tes= missForest(test[,c(1,7:13)])

df_imp_tra= df_imp_tra$ximp
df_imp_tes= df_imp_tes$ximp

df_train=data.frame(train[,c(2,3,4,5,6,14)],df_imp_tra)
df_test=data.frame(test[,c(2,3,4,5,6)],df_imp_tes)


## Lets impute totals again
## then we do impute types
## then na omit the legendrary
## after wardss we can probably fit a model .
## also one hot coding 
## we also could not impute for Type1. as there are 53 categories
## but few observations , dont have enough data, since its missing at random
## we could kinda impute it randomly .
for ( i in 1:nrow(df_train)){
  if(is.na(df_train$Total[i])){
    if(!any(is.na(df_train[i,8:13]))){
      df_train[i,5]=sum(df_train[i,8:13])
    }
  }
}
for ( i in 1:nrow(df_test)){
  if(is.na(df_test$Total[i])){
    if(!any(is.na(df_test[i,7:12]))){
      df_test[i,5]=sum(df_test[i,7:12])
    }
  }
}

## now only resposne should have na

gg_miss_var(df_train)
##legendrary imputation, if  mega and total 780 or more, 
## set to not mega
for(i in 1:nrow(df_test)){
  
    if(df_test$Total[i]>=779){
      df_test$X[i]=0
  }
}


## it seems like X. is about forms, where non mega and mega version have same value
## can impute legendrary war here, like if its simiiliar is legendrary, it also is 
## first for train in train
for(i in 1:nrow(df_train)){
  for ( j in (i+1):nrow(df_train)){
    if(!is.na(df_train$X.[i])&!is.na(df_train$X.[j])){
    if(df_train$X.[i]==df_train$X.[j]){
      if(!is.na(df_train$Legendary[j])){
      if(df_train$Legendary[j]==T){
        df_train$Total[i]=10000 ##should convince the ml model 
        df_train$Total[j]=10000
        df_train$X[j]=0
      }
      }
    }
    }
  }
}


for(i in 1:nrow(df_train)){
  for ( j in 1:nrow(df_test)){
    if(!is.na(df_train$X.[i])&!is.na(df_test$X.[j])){
      if(df_train$X.[i]==df_test$X.[j]){
        if(!is.na(df_train$Legendary[i])){
          if(df_train$Legendary[i]==T){
            df_test$Total[j]=10000 ##should convince the ml model
            df_test$X[j]=0
          }
        }
      }
    }
  }
}
## Now we did a fun feature engineering, where if other version is mega, check non mega,
## however probably could instead just check in ID, lets do that, some have non megas with 2 ids
## lets impute in test data, this time for non megas, where if X. is same, then check legendrary
## then if not, set its X to 0, and Total to 0, feature engineering is everything 
maddex= df_test$X.[df_test$X. %in% df_train$X.]

for (i in 1:length(maddex)){
  temp_var= maddex[i]
  print(i)
  temp_df= df_train[df_train$X.==temp_var,]
  temp_df=na.omit(temp_df)
  if(nrow(temp_df)!=0){ 
    if((!temp_df$Legendary[1])){
    df_test$X[df_test$X.==temp_var]=FALSE ## Remove mega attribute if any 
    df_test$Total[df_test$X.==temp_var]=FALSE ## reduce stats if many
    }
  }
}


### we have yet to from initial analysis do the dragon adjustment, where dragons like megas
### have inflated statistics .  once we do it, i dont see what else engineering can be done
#for (i in 1:nrow(df_test)){
#  if(df_test$Type.1[i]=="Dragon"|df_test$Type.2[i]=="Dragon"){
#    df_test$Total[i]=df_test$Total[i]-35
#    
#  }
#}
#### Also usually only one type pokemons are less likely, example in our test was slaking so 
#### engineer for it too and try for 8th time
#for (i in 1:nrow(df_test)){
#  if(df_test$Type.2[i]=="none"){
#    df_test$Total[i]=df_test$Total[i]-35
#    
#  }
#}
##### Was useless, so deleted. 

gg_miss_var(df_train)
df_test=df_test[,2:13]
df_train=df_train[,2:14]

train_df=na.omit(df_train)
index=sample(1:nrow(train_df),nrow(train_df)*0.8,replace=F)

tr_df=train_df[index,]
te_df=train_df[-index,]

fit= rpart(Legendary~.,data = tr_df[,-c(1,2,3)], method = 'class')

rpart.plot(fit, type = 1,fallen.leaves=FALSE)
### if u here get error resample , from line 257
predict_unseen <-predict(fit, te_df[,-c(1,2,3)], type = 'class')
confusionMatrix(as.factor(predict_unseen),as.factor(te_df$Legendary))

###hmm

predict_unseen <-predict(fit, df_test[,-c(1,2,3)], type = 'class')
predict_unseen
data=cbind(predict_unseen)
data[data==1]= "FALSE"
data[data==2]= "TRUE"
write.csv(data,"data.csv")

##probably two stage prediction is better 
## randomization algorithms seem make everything worse when you have such discrete
## rules in the data, so we also wont use them 
library(xgboost)
x_train=data.matrix(tr_df[,c(-1,-2,-3,-5)])
y_train=tr_df[,5]


x_test=data.matrix(te_df[,c(-1,-2,-3,-5)])
y_test=te_df[,5]

xgboost_train = xgb.DMatrix(data=x_train, label=y_train)
xgboost_test = xgb.DMatrix(data=x_test, label=y_test)

model <- xgboost(data = xgboost_train,                    # the data   
                 max.depth=4,                          # max depth 
                 nrounds=75)                              # max number of boosting iterations

pred_test = predict(model, xgboost_test)
pred_test

pred_y_logical=c()
for (i in 1:71){
  if(pred_test[i]>0.445){
    pred_y_logical=append(pred_y_logical,TRUE)
  }
  else{
    pred_y_logical=append(pred_y_logical,FALSE)
  }
}
pred_y_logical=as.factor(pred_y_logical)
y_test=as.factor(y_test)
confusionMatrix(y_test, pred_y_logical)
#### looks fair, lets apply to df_test
df_test[is.na(df_test$X),5]=1

x_test=data.matrix(df_test[,c(-1,-2,-3)])
#y_test=df_test[,5]
xgboost_test = xgb.DMatrix(data=x_test)#, label=y_test)
pred_test = predict(model, xgboost_test)
pred_test

pred_y_logical=c()
for (i in 1:length(pred_test)){
  if(pred_test[i]>0.445){
    pred_y_logical=append(pred_y_logical,TRUE)
  }
  else{
    pred_y_logical=append(pred_y_logical,FALSE)
  }
}
pred_y_logical=as.factor(pred_y_logical)
xgdata=cbind(pred_y_logical)
write.csv(xgdata,"xgdata.csv")

##We can do final iteration some var'ables had eqauivalent non mega in test predicted
## but they are unsued  to fix  just another for loop which no machine learning model writes
##Can fix our proble m

df_temp_test=cbind.data.frame(df_test,xgdata,test$X.)

df_temp_test$`test$X.`


for(i in 1:nrow(df_temp_test)){
  for(j in 1:nrow(df_temp_test[-i,])){
    if(df_temp_test$`test$X.`[i]==df_temp_test[-i,]$`test$X.`[j]){
      if(df_temp_test[-i,]$pred_y_logical[j]==2){
        df_temp_test$pred_y_logical[i]=2
        print("i")
      }
    }
  }
}
df_temp_test$pred_y_logical

df_temp_test$pred_y_logical[df_temp_test$pred_y_logical==1]= "FALSE"
df_temp_test$pred_y_logical[df_temp_test$pred_y_logical==2]= "TRUE"

write.csv(df_temp_test$pred_y_logical,"one_more_light")
