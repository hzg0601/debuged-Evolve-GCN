models=(gcn skipgcn skipfeatsgcn egcn_o egcn_h)
file='./experiments/parameters_example.yaml'
i=0
for model in ${models[@]};
do  
    echo currently processing model is: $model
    sed -i "21s/^model:.*$/model: ${model}/g" $file # 将第21行model开头的全部内容替换为model: $model

    nohup python run_exp.py --config_file $file > $model".log" 2>&1 &
    echo the id of current process is:  $!
    
    wait $!
    let i++
done

