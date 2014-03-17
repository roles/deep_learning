#!/bin/sh

#./cross_valid_partition.sh -l cross_valid/label.out 
#                           -d cross_valid/data.out -i cross_valid/line.index -o .

index_file="line.index"
data_file="data.out"
label_file="label.out"
nfold=10
data_file_prefix=2
label_file_prefix=1

while getopts ":l:d:i:o:n:" opt
do
    case $opt in
        d) data_file=$OPTARG;;
        l) label_file=$OPTARG;;
        i) index_file=$OPTARG;;
        o) output_folder=$OPTARG;;
        n) nfold=$OPTARG;;
        ?) echo "invalid parameter"; exit 1;;
    esac
done

if [ ! -d "$output_folder/cross_valid" ];
then
    mkdir $output_folder/cross_valid
fi

nfeature=`sed '2q;d' $data_file`
nline=`sed '1q;d' $data_file`
nlabel=`sed '1q;d' $label_file`
fold_size=$((($nline-1)/$nfold+1))

for i in `seq $nfold`
do
    if [ ! -d "$output_folder/cross_valid/fold_$i" ];
    then
        mkdir $output_folder/cross_valid/fold_$i
    fi
    fold_train_data_file=$output_folder/cross_valid/fold_$i/train_data.txt
    fold_train_label_file=$output_folder/cross_valid/fold_$i/train_label.txt
    fold_valid_data_file=$output_folder/cross_valid/fold_$i/valid_data.txt
    fold_valid_label_file=$output_folder/cross_valid/fold_$i/valid_label.txt

    if [ $i -eq $nfold ];
    then
        fold_nline=$(($nline - ($nfold - 1)*$fold_size))   
    else
        fold_nline=$fold_size
    fi

    echo $fold_nline > $fold_valid_data_file
    echo $nfeature >> $fold_valid_data_file
    echo $nlabel > $fold_valid_label_file

    echo $(($nline - $fold_nline)) > $fold_train_data_file
    echo $nfeature >> $fold_train_data_file
    echo $nlabel > $fold_train_label_file

    fold_begin=$((($i-1)*$fold_size+1))
    fold_end=$(($i*$fold_size))
    if [ $i -ne 1 ];
    then
        sed -n "$(($data_file_prefix+1)),$(($fold_begin+$data_file_prefix-1))p" $data_file >> $fold_train_data_file 
        sed -n "$(($label_file_prefix+1)),$(($fold_begin+$label_file_prefix-1))p" $label_file >> $fold_train_label_file 
    fi
    if [ $i -ne $nfold ];
    then
        sed -n "$(($data_file_prefix+$fold_end+1)),$(($nline+$data_file_prefix))p" $data_file >> $fold_train_data_file 
        sed -n "$(($label_file_prefix+$fold_end+1)),$(($nline+$label_file_prefix))p" $label_file >> $fold_train_label_file 
    fi
    sed -n "$(($data_file_prefix+$fold_begin)),$(($data_file_prefix+$fold_end))p" $data_file >> $fold_valid_data_file
    sed -n "$(($label_file_prefix+$fold_begin)),$(($label_file_prefix+$fold_end))p" $label_file >> $fold_valid_label_file
done
