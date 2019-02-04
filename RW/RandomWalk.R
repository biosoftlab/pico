#Simulate Random Walk Algorithm
#Input parameters for restart rate and threshold value
options(scipen=999)
args=(commandArgs(TRUE))

network_file = args[1]
network_dict = args[2]
drug_file = args[3]
output_file = args[4]

r=as.numeric(0.7)
threshold=as.numeric(0.0001)

strDrugTargetDicFile = paste(network_dict)
strNetworkFile = paste(network_file)
strDrugTargetInteractionFile = paste(drug_file)

DBName = strsplit(strNetworkFile, ".txt")[[1]]
DrugTargetDic <- read.delim( strDrugTargetDicFile, header=F )
Network_size = dim(DrugTargetDic)[1]

w = mat.or.vec( Network_size, Network_size )

Model <- read.delim( strNetworkFile, header = F )
x = Model[1]
y = Model[2]
EdgeWeight = Model[3]

N = dim(Model[2])[1]

for( i in 1:N ){
  x[i,]
  y[i,]
  w[x[i,],y[i,]] = as.numeric( EdgeWeight[i,] )
}

someenv<-new.env()
ModelTarget <- read.delim( strDrugTargetInteractionFile, header = F )
N_Target = dim( ModelTarget[2] )[1]
x_Target = ModelTarget[1] #drug
y_Target = ModelTarget[2] #target
for(i in 1:N_Target){
  if(!is.na( y_Target[i,]))  {
    someenv[[toString( x_Target[i,] )]] = append( someenv[[toString( x_Target[i,] )]], y_Target[i,] )
  }
}
ls(someenv)
keys = ls( someenv )

for( i in 1:length(keys)){
  p0 = t( mat.or.vec( 1, Network_size ) )
  for( j in 1:length( someenv[[keys[i]]] ) ){
    p0[someenv[[keys[i]]][j]] = 1
  }
  temporal_result = p0
  
  p = p0
  MatrixVariationValue = 1
  index = 0

  while( MatrixVariationValue > threshold ){
    before_p = p
    p = ( ( 1 - r ) * t( w ) ) %*% p + r * p0
    after_p = p
    index = index + 1

    temporal_result <- cbind(temporal_result, after_p)

    MatrixVariationValue = sum( abs( before_p - after_p ) )
    print(paste(toString(index), 'th iter, matrixVariation Value:', toString(MatrixVariationValue)))

    # rownames( after_p ) <- DrugTargetDic$V1
    # write.table( after_p, file = paste('results/',foldername,"/TempRW/tmp_", toString(index), "_RandomWalk_Result.txt", sep = ""), row.names = TRUE, col.names = FALSE, sep = "\t", quote = FALSE, eol = "\r\n")
  }
  #rownames(temporal_result) <- DrugTargetDic$V1
  #write.table( temporal_result, file = paste(output_file), row.names = TRUE, col.names = FALSE, sep = "\t", quote = FALSE, eol = "\r\n")

  rownames( after_p ) <- DrugTargetDic$V1
  write.table( after_p, file = paste(output_file), row.names = TRUE, col.names = FALSE, sep = "\t", quote = FALSE, eol = "\r\n")
}