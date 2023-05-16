models=(dt rf mlp lr)
features=(AF LF NE AF+NE LF+NE)
nes=(egcn_h egcn_o skipfeatsgcn gcn skipgcn)

file='./experiments/parameters_example.yaml'
i=0
for model in ${models[@]};
do  
    for feature in ${features[@]};
    do
        for ne in ${nes[@]};
        do
            echo currently processing model, feature,ne is: $model,${feature},${ne}
            sed -i "90s/^\smodel: .*$/\tmodel: ${model}/g" $file
            sed -i "91s/^\sfeature: .*$/\tfeature: ${feature}/g" $file
            sed -i "92s/^\sne: .*$/\tne: ${ne}/g" $file
            nohup python ml_models.py --config_file $file > "./log/log3_${model}_${feature}_${ne}.log" 2>&1 &
            echo the id of current process is: $!
            wait $!
            let i++
        done
    done
done
