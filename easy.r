library("Rcpp")
library("volesti")
x<-5:8
res<-5:8
i = 1
for (e in x){
P = GenRandVpoly(3, e)
res[i] = volume(P)
i=i+1
}
plot(res)
