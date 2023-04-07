yaml_file=(parameters_elliptic_egcn_h.yaml parameters_elliptic_egcn_o.yaml)
log_file=(elliptic_egcn_h.log elliptic_egcn_o.log)
i=0
for loop in ${#yaml_file[@]}; 
do
    nohup python run_exp.py --config_file ./experiments/${yaml_file[i]} > ${log_file[i]} 2>&1 & 
    echo the id of current process is: $!
    echo the current loop is: ${yaml_file[i]}
    wait $!
    let i++
done
