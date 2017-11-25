library(readr)
library(cowplot)

which_model = "All_30_noPosNoChunk_MultiSkip512_30"
#-----------------------Loss-----------------
loss <- read_delim("/Users/Cristhian/Documents/ThesisNER/StatisticalNER/Spanish_noPosNoChunk_30_loss.txt", 
                   delim = "\t", 
                   col_names = FALSE)
loss <- as.numeric(loss)
loss<-(t(t(loss)))
loss <- as.data.frame(loss)
length(loss$V1)
l <- ggplot(data = loss, aes(y= V1, x=seq(from = 1, to = length(loss$V1)), group=1)) +
  geom_line(color="blue", size=1)+
  geom_point(color="blue") +
  #geom_text(data = loss,label=loss$V1)+
  labs(x = "Epochs", y = "Loss", title = "Spanish Original 30 loss")
#-----------------------F1-----------------
f1 <- read_delim("/Users/Cristhian/Documents/ThesisNER/StatisticalNER/Spanish_noPosNoChunk_30_f1.txt", 
                   delim = "\t", 
                   col_names = FALSE)
f1 <- as.numeric(f1)
f1<-(t(t(f1)))
f1 <- as.data.frame(f1)
f1p<- ggplot(data = f1, aes(y= V1, x=seq(from = 1, to = length(f1$V1)), group=1)) +
  geom_line(color="blue", size=1)+
  geom_point(color="blue") +
  labs(x = "Epochs", y = "F1 - Score", title = "Spanish Original 30 F1")
#-----------------------Recall-----------------
recall <- read_delim("/Users/Cristhian/Documents/ThesisNER/StatisticalNER/Spanish_noPosNoChunk_30_recall.txt", 
                   delim = "\t", 
                   col_names = FALSE)
recall <- as.numeric(recall)
recall<-(t(t(recall)))
recall <- as.data.frame(recall)
recallq <-ggplot(data = recall, aes(y= V1, x=seq(from = 1, to = 101), group=1)) +
  geom_line(color="blue", size=1)+
  geom_point(color="blue") +
  labs(x = "Epochs", y = "Recall", title = "Spanish Original 30 Recall")

#-----------------------Precision-----------------
precision <- read_delim("/Users/Cristhian/Documents/ThesisNER/StatisticalNER/Spanish_noPosNoChunk_30_precision.txt", 
                   delim = "\t", 
                   col_names = FALSE)
precision <- as.numeric(precision)
precision<-(t(t(precision)))
precision <- as.data.frame(precision)
length(precision$V1)
precisionp<-ggplot(data = precision, aes(y= V1, x=seq(from = 1, to = length(precision$V1)), group=1)) +
  geom_line(color="blue", size=1)+
  geom_point(color="blue") +
  labs(x = "Epochs", y = "Precision", title = "Spanish Original 30 Precision")


#-----------------------Spanish-----------------
ggdraw() +
  draw_plot(l, x = 0, y = .5, width = .5, height = .5) +
  draw_plot(f1p, x = .5, y = .5, width = .5, height = .5) +
  draw_plot(recallq, x = 0, y = 0, width = .5, height = .5) +
  draw_plot(precisionp, x = .5, y = 0, width = .5, height = .5) 
  





#-----------------------English-----------------

#-----------------------Loss-----------------
loss <- read_delim("/Users/Cristhian/Documents/ThesisNER/StatisticalNER/English_noPosNoChunk_30_loss.txt", 
                   delim = "\t", 
                   col_names = FALSE)
loss <- as.numeric(loss)
loss<-(t(t(loss)))
loss <- as.data.frame(loss)
length(loss$V1)
l <- ggplot(data = loss, aes(y= V1, x=seq(from = 1, to = length(loss$V1)), group=1)) +
  geom_line(color="blue", size=1)+
  geom_point(color="blue") +
  #geom_text(data = loss,label=loss$V1)+
  labs(x = "Epochs", y = "Loss", title = "English Original 30 loss")
#-----------------------F1-----------------
f1 <- read_delim("/Users/Cristhian/Documents/ThesisNER/StatisticalNER/English_noPosNoChunk_30_f1.txt", 
                 delim = "\t", 
                 col_names = FALSE)
f1 <- as.numeric(f1)
f1<-(t(t(f1)))
f1 <- as.data.frame(f1)
f1p<- ggplot(data = f1, aes(y= V1, x=seq(from = 1, to = length(f1$V1)), group=1)) +
  geom_line(color="blue", size=1)+
  geom_point(color="blue") +
  labs(x = "Epochs", y = "F1 - Score", title = "English Original 30 F1")
#-----------------------Recall-----------------
recall <- read_delim("/Users/Cristhian/Documents/ThesisNER/StatisticalNER/Spanish_noPosNoChunk_30_recall.txt", 
                     delim = "\t", 
                     col_names = FALSE)
recall <- as.numeric(recall)
recall<-(t(t(recall)))
recall <- as.data.frame(recall)
recallq <-ggplot(data = recall, aes(y= V1, x=seq(from = 1, to = 101), group=1)) +
  geom_line(color="blue", size=1)+
  geom_point(color="blue") +
  labs(x = "Epochs", y = "Recall", title = "English Original 30 Recall")

#-----------------------Precision-----------------
precision <- read_delim("/Users/Cristhian/Documents/ThesisNER/StatisticalNER/Spanish_noPosNoChunk_30_precision.txt", 
                        delim = "\t", 
                        col_names = FALSE)
precision <- as.numeric(precision)
precision<-(t(t(precision)))
precision <- as.data.frame(precision)
length(precision$V1)
precisionp<-ggplot(data = precision, aes(y= V1, x=seq(from = 1, to = length(precision$V1)), group=1)) +
  geom_line(color="blue", size=1)+
  geom_point(color="blue") +
  labs(x = "Epochs", y = "Precision", title = "English Original 30 Precision")


#-----------------------English-----------------
ggdraw() +
  draw_plot(l, x = 0, y = .5, width = .5, height = .5) +
  draw_plot(f1p, x = .5, y = .5, width = .5, height = .5) +
  draw_plot(recallq, x = 0, y = 0, width = .5, height = .5) +
  draw_plot(precisionp, x = .5, y = 0, width = .5, height = .5) 






#-----------------------German-----------------

#-----------------------Loss-----------------
loss <- read_delim("/Users/Cristhian/Documents/ThesisNER/StatisticalNER/German_noPosNoChunk_30_loss.txt", 
                   delim = "\t", 
                   col_names = FALSE)
loss <- as.numeric(loss)
loss<-(t(t(loss)))
loss <- as.data.frame(loss)
length(loss$V1)
l <- ggplot(data = loss, aes(y= V1, x=seq(from = 1, to = length(loss$V1)), group=1)) +
  geom_line(color="blue", size=1)+
  geom_point(color="blue") +
  #geom_text(data = loss,label=loss$V1)+
  labs(x = "Epochs", y = "Loss", title = "German Original 30 loss")
#-----------------------F1-----------------
f1 <- read_delim("/Users/Cristhian/Documents/ThesisNER/StatisticalNER/German_noPosNoChunk_30_f1.txt", 
                 delim = "\t", 
                 col_names = FALSE)
f1 <- as.numeric(f1)
f1<-(t(t(f1)))
f1 <- as.data.frame(f1)
f1p<- ggplot(data = f1, aes(y= V1, x=seq(from = 1, to = length(f1$V1)), group=1)) +
  geom_line(color="blue", size=1)+
  geom_point(color="blue") +
  labs(x = "Epochs", y = "F1 - Score", title = "German Original 30 F1")
#-----------------------Recall-----------------
recall <- read_delim("/Users/Cristhian/Documents/ThesisNER/StatisticalNER/German_noPosNoChunk_30_recall.txt", 
                     delim = "\t", 
                     col_names = FALSE)
recall <- as.numeric(recall)
recall<-(t(t(recall)))
recall <- as.data.frame(recall)
recallq <-ggplot(data = recall, aes(y= V1, x=seq(from = 1, to = 101), group=1)) +
  geom_line(color="blue", size=1)+
  geom_point(color="blue") +
  labs(x = "Epochs", y = "Recall", title = "German Original 30 Recall")

#-----------------------Precision-----------------
precision <- read_delim("/Users/Cristhian/Documents/ThesisNER/StatisticalNER/Geram_noPosNoChunk_30_precision.txt", 
                        delim = "\t", 
                        col_names = FALSE)
precision <- as.numeric(precision)
precision<-(t(t(precision)))
precision <- as.data.frame(precision)
length(precision$V1)
precisionp<-ggplot(data = precision, aes(y= V1, x=seq(from = 1, to = length(precision$V1)), group=1)) +
  geom_line(color="blue", size=1)+
  geom_point(color="blue") +
  labs(x = "Epochs", y = "Precision", title = "German Original 30 Precision")


#-----------------------German-----------------
ggdraw() +
  draw_plot(l, x = 0, y = .5, width = .5, height = .5) +
  draw_plot(f1p, x = .5, y = .5, width = .5, height = .5) +
  draw_plot(recallq, x = 0, y = 0, width = .5, height = .5) +
  draw_plot(precisionp, x = .5, y = 0, width = .5, height = .5) 


  