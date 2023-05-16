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
            

            echo currently processing model, feature,ne is: ${model},${feature},${ne}
            sed -i "91s/[\t|(  )]model: .*$/ model: ${model}/g" $file
            sed -i "93s/[\t|(  )]feature: .*$/ feature: ${feature}/g" $file
            sed -i "95s/[\t|(  )]ne: .*$/ ne: ${ne}/g" $file
            nohup python ml_models.py --config_file $file >> ./log/log3_ml.log 2>&1 & #"./log/log3_${model}_${feature}_${ne}.log" 2>&1 &
            echo the id of current process is: $!
            wait $!
            let i++
            # 字符串 用== 数字 -eq
            if ( [ $feature == AF ] || [ $feature == LF ] );then
                echo jump out of current loop of nes when feature equals to AF or LF.
                continue 2
            fi
        done
    done
done
