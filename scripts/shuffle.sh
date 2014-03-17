#!/bin/sh

#./shuffle.sh -d train.data.format -l train.label.format -o cross_valid

data_file_prefix=2
label_file_prefix=1

while getopts ":l:d:o:" opt
do
    case $opt in
        d) data_file=$OPTARG;;
        l) label_file=$OPTARG;;
        o) output_folder=$OPTARG;;
        ?) echo "invalid parameter"; exit 1;;
    esac
done

line_count=$((`wc -l $data_file | awk '{print $1}'` - $data_file_prefix))
echo $line_count

sed -n "1,${data_file_prefix}p" $data_file > $output_folder/data.out
sed -n "1,${label_file_prefix}p" $label_file > $output_folder/label.out

seq $line_count | sort -R | while read line
do
    echo $line >> $output_folder/line.index
    sed "$(($line + $data_file_prefix))q;d" $data_file >> $output_folder/data.out
    sed "$(($line + $label_file_prefix))q;d" $label_file >> $output_folder/label.out
done
