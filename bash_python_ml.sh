models=(dt rf mlp lr)
features=(AF LF NE AF+NE LF+NE)
nes=(egcn_h egcn_o skipfeatsgcn gcn skipgcn)

file='./experiments/parameters_example.yaml'
i=0
for model in ${models[@]};
do  
    echo currently processing model, feature,ne is: $model,${features[i]},${nes[i]}
    sed -i "90s/^model:.*$/model: ${model}/g" $file
    sed -i "91s/^feature:.*$/feature: ${features[i]}/g" $file
    sed -i "92s/^ne:.*$/ne: ${nes[i]}/g" $file
    nohup python ml_models.py --config_file $file > "./log/log3_${model}_${features[i]}_${nes[i]}.log" 2>&1 &
    echo the id of current process is: $!
    wait $!
    let i++

done
