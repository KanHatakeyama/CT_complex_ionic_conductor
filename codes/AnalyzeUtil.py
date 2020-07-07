from sklearn.metrics import r2_score
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#columns to be analyzed
selected_columns=[
            "salt","A_LUMO","D_HOMO", "HOMO_LUMO_gap", "thickness(um)",
            "s",
            'A_SlogP','A_TPSA',
            'D_SlogP','D_TPSA',
            #***ignore less important descriptors**
            #'A_AMW',
            #'A_NumRotatableBonds', 'A_NumHBD', 'A_NumHBA', 
            #'A_NumAmideBonds',
            #'A_NumHeteroAtoms', 'A_NumHeavyAtoms', 'A_NumAtoms',
            #'A_NumStereocenters', 'A_NumRings', 'A_NumAromaticRings',
            #'A_NumSaturatedRings', 'A_NumAliphaticRings',
            #'A_NumAromaticHeterocycles', 'A_NumSaturatedHeterocycles',
            #'A_NumAliphaticHeterocycles', 'A_NumAromaticCarbocycles',
            #'A_NumSaturatedCarbocycles', 'A_NumAliphaticCarbocycles',
            # 'D_AMW', 
            #'D_NumRotatableBonds', 'D_NumHBD', 'D_NumHBA',
            #'D_NumAmideBonds', 'D_NumHeteroAtoms', 'D_NumHeavyAtoms', 'D_NumAtoms',
            #'D_NumStereocenters', 'D_NumRings', 'D_NumAromaticRings',
            #'D_NumSaturatedRings', 'D_NumAliphaticRings',
            #'D_NumAromaticHeterocycles', 'D_NumSaturatedHeterocycles',
            #'D_NumAliphaticHeterocycles', 'D_NumAromaticCarbocycles',
            #'D_NumSaturatedCarbocycles', 'D_NumAliphaticCarbocycles'
         ]


#automatically analyze parameter dependencies by elastic net
def get_coeff_list(target,df,model,R2_mode=False):
    """
    target: target parameter name (str) to be predicted
    df: dataframe used for analysis
    alpha: regularization parameter in elastic net
    return: coefficient list
    """
    x_names=set(list(df.columns))-set(target)

    y=df[target]
    x=df[x_names]

    model.fit(x, y)
    R2=r2_score(y, model.predict(x))
    
    coeff_list=[]
    for num,i in enumerate(x_names):
        if R2_mode:
            coeff=R2
        else:
            coeff=R2*abs(model.coeff(num))
        coeff_list.append((i,target[0],coeff))
        
    return coeff_list



    
#draw graph
def to_graph(all_coeff_list,param_list,arrow_scale=30,coeff_cutoff=0.1):    
    """
    all_coeff_list: coefficient list
    param_list: list of parameter names
    arrow_scale:scale of arrows drawn
    coeff_cutoff: cuttoff value for drawing edges

    return: graph image, networkx graph objext
    """
    #prepare graph
    g = nx.MultiDiGraph()
    g.add_nodes_from(param_list)

    for i in all_coeff_list:
        node_from, node_to, coeff = i[0],i[1],i[2]

        coeff=coeff*arrow_scale
        if abs(coeff)>coeff_cutoff:
            g.add_edge(node_from,node_to,
                       color='red',
                       penwidth =abs(coeff)         
                      )

    #delete isolated nodes
    node_list=list(g.nodes)
    for node in node_list:
        if len(list(g.neighbors(node)))==0:
            g.remove_node(node)
            print("removed: ",node)
    
    agraph = nx.nx_agraph.to_agraph(g)
    img = agraph.draw(prog="dot", format="svg")
    return img,g


#convert coeff list to dataframe
def coeff_list_to_df(all_coeff_list):
    """
    all_coeff_list: coefficient list
    return dataframe
    """
    param_list=list(set(list(zip(*all_coeff_list))[0]))
    param_list.sort()
    coeff_mat=np.zeros((len(param_list),len(param_list)))

    param_dict={}
    for num,i in enumerate(param_list):
        param_dict[i]=num

    for i in all_coeff_list:
        y=param_dict[i[0]]
        x=param_dict[i[1]]
        coeff_mat[x][y]=i[2]
        
    heat_df= pd.DataFrame(data=coeff_mat, index=param_list,columns=param_list)

    return heat_df

# draw heatmap for a coefficient list
def heatmap(all_coeff_list):
    df=coeff_list_to_df(all_coeff_list)
    plt.figure(figsize=(10, 10))
    sns.heatmap(df,cmap='Reds')


#wrapper functions to extraxt coefficients  

#elastic net
class ENWrapper:
    def __init__(self,model):
        self.model=model
    
    def predict(self,x):
        return self.model.predict(x)
    
    def fit(self,x,y):
        self.model.fit(x,y)
        
    def coeff(self,num):
        return self.model.coef_[num]
    
#random forest    
class RFRWrapper(ENWrapper):
    def coeff(self,num):
        return self.model.feature_importances_[num]
