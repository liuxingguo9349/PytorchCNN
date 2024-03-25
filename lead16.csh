#!/bin/csh
# c-shell script for heatmap analysis.

# set lead month 月份
foreach LEAD (16)  #leadmonth 1-8  4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#setenv LEAD 1

setenv BBB 400                          #batch size cmip
setenv BST 20                           #batch size transfer

# set target season
@ TMON = 7
while ($TMON <= 12) 

setenv HHH '/home'						# Main directory
setenv RET 'ssthenlstest1'				    # train directory

setenv MOD $LEAD'mon'$TMON			    # 1mon1
setenv TYP 'cmip'						#type cmip
setenv TYPE 'transfer'					#type transfer

setenv CUDA 'cuda:4'

echo $MOD													

foreach conf ( 30 50 )      # Number of conv. features
foreach hidf ( 30 50 )      # Number of hidden neurons


setenv opname 'C'$conf'H'$hidf

echo $opname		#C30H30

@ ens = 1
while ($ens <= 10)									


mkdir -p $HHH/ln/$RET/allretrainresulttrans
mkdir -p $HHH/ln/$RET/allretrainresultcmip
mkdir -p $HHH/ln/$RET/$MOD/$TYP/$opname
mkdir -p $HHH/ln/$RET/$MOD/$TYP/$opname/EN$ens			#mnt/retrainconcat/1mon1/cmip/C30H30/0.1
mkdir -p $HHH/ln/$RET/$MOD/$TYP/src/$opname


mkdir -p $HHH/ln/$RET/$MOD/$TYPE/$opname
mkdir -p $HHH/ln/$RET/$MOD/$TYPE/$opname/EN$ens			#mnt/retrainconcat/1mon1/transfer/C30H30/0.1
mkdir -p $HHH/ln/$RET/$MOD/$TYPE/src/$opname

# 训练 cmip

cd $HHH/ln/$RET/$MOD/$TYP/src/$opname						# 1mon1/cmip/src/c30h30
cp -f $HHH/ln/$RET/cmip.sample .	
cp -f $HHH/ln/$RET/model.py .				

# make layer
sed "s/chlist/$opname/g"							cmip.sample > tmp1
sed "s/lead_mon/$LEAD/g"								 tmp1 > tmp2
sed "s/target_mon/$TMON/g"								 tmp2 > tmp1
sed "s/convfilter/$conf/g"								 tmp1 > tmp2         
sed "s/hiddfilter/$hidf/g"								 tmp2 > tmp1
sed "s/lmont/$MOD/g"									 tmp1 > tmp2
sed "s/document/$RET/g"									 tmp2 > tmp1
sed "s/cudanum/$CUDA/g"                                  tmp1 > tmp2
sed "s/number/$ens/g"								    tmp2 > cmip.py

python cmip.py

# 训练transfer

cd $HHH/ln/$RET/$MOD/$TYPE/src/$opname						# 1mon1/transfer/src/c30h30
cp -f $HHH/ln/$RET/transfer.sample .			
cp -f $HHH/ln/$RET/model.py .

# make layer
sed "s/chlist/$opname/g"							transfer.sample > tmpt1
sed "s/lead_mon/$LEAD/g"                                 tmpt1 > tmpt2
sed "s/target_mon/$TMON/g"                               tmpt2 > tmpt1
sed "s/convfilter/$conf/g"								 tmpt1 > tmpt2
sed "s/hiddfilter/$hidf/g"								 tmpt2 > tmpt1
sed "s/lmont/$MOD/g"									 tmpt1 > tmpt2
sed "s/document/$RET/g"									 tmpt2 > tmpt1
sed "s/cudanum/$CUDA/g"                                  tmpt1 > tmpt2
sed "s/number/$ens/g"								    tmpt2 > transfer.py

python transfer.py

@ ens = $ens + 1									

end   #while ens


# 预测predict

cd $HHH/ln/$RET/$MOD/$TYPE/src/$opname						# ln/CNNcsh/lmont/transfer/src/c30h30
cp -f $HHH/ln/$RET/predict.sample .			
cp -f $HHH/ln/$RET/model.py .

# make layer
sed "s/chlist/$opname/g"							predict.sample > tmpv1
sed "s/lead_mon/$LEAD/g"                                 tmpv1 > tmpv2
sed "s/target_mon/$TMON/g"                               tmpv2 > tmpv1
sed "s/convfilter/$conf/g"								 tmpv1 > tmpv2
sed "s/hiddfilter/$hidf/g"								 tmpv2 > tmpv1
sed "s/lmont/$MOD/g"									 tmpv1 > tmpv2
sed "s/document/$RET/g"									 tmpv2 > tmpv1
sed "s/cudanum/$CUDA/g"                                  tmpv1 > tmpv2
sed "s/number/$ens/g"								    tmpv2 > predict.py

python predict.py


end    #foreach hidf
end	   #foreach conf



@ TMON = $TMON + 1

end		# while TMON

end     #foreach LEAD

