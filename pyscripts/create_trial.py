#! /usr/bin/env python
import yaml
import os

def make_yaml(N, labels, feature, k, M, split):

    struct = {}

    struct["data"]={"labels":labels, "N":N}

    if feature == "scatter":
        struct["features"] = {'scatter': {'M':M, 'N':N,'T': 4*N}}
    elif feature == "holder":
        struct["features"]="holder"

    ml = {}

    ml["preprocess"]={'training_split': split}
    ml["classification"]={"knn":{"k":5}}

    struct["machine_learning"] = ml

    return struct


def main():

    output_dir = '../experiments'
    labels = ['Ordovician', 'BlackRiver', 'Utica', 'Kope', 'TL']

    k_ = [3,5,7,10]

    N_=[64,128,256]

    M_ = [[0],[0,1], [0,1,2]]

    training_split = [.5,.75]

    features = ["scatter", "holder"]
    
    i = 0
    for M in M_:

        for feature in features:
     
            for k in k_:
         
                for N in N_:
             
                    for split in training_split:
                    
                        struct = make_yaml(N, labels, feature, k, M, split)

                        outfile = os.path.join(output_dir, 'exp'+str(i)+'.yaml')

                        with open(outfile, 'w') as f:
                            f.write(yaml.dump(struct, default_flow_style=False))
                            i+=1

                
                
    
if __name__ == "__main__":

    main()

