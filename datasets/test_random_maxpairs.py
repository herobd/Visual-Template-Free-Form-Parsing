from datasets.random_maxpairs import RandomMaxPairsDataset
from datasets.random_maxpairs import collate
import networkx as nx
from matplotlib import pyplot as plt
import torch


def display(input,output=None):
    features = input[0]
    edges = input[1].t().tolist()
    gt = input[2]

    
    G=nx.Graph()
    G.add_nodes_from(list(range(features.size(0))))

    G.add_edges_from(edges)

    rename={}
    senders=[]
    recievers=[]
    others=[]
    for i in range(features.size(0)):
        rename[i] = '{:.2}'.format(features[i,0])
        #    rename[i] = ' '
    #print(output)
    #G=nx.relabel_nodes(G,rename)

    edgesTrue = input[1].t()[gt[:,0]].tolist()
    edgesFalse = input[1].t()[1-gt[:,0]].tolist()
    if output is not None:
        edgeNamesC={}
        edgeNamesI={}
        for i in range(len(edges)):
            if (gt[i] and output[i,0]>0.5) or (not gt[i] and output[i,0]<0.5):
                edgeNamesC[tuple(edges[i])] = '{:.2}'.format(output[i,0])
            else:
                edgeNamesI[tuple(edges[i])] = '{:.2}'.format(output[i,0])


    #nx.draw(G)
    #plt.show()
        
    pos=nx.spring_layout(G) # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G,pos,
                           nodelist=list(range(features.size(0))),
                           node_color='yellow',
                           node_size=100,
                       alpha=0.8)

    # edges
    #nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
    nx.draw_networkx_edges(G,pos,
                           edgelist=edgesFalse,
                           width=5,alpha=0.6,edge_color='grey')
    nx.draw_networkx_edges(G,pos,
                           edgelist=edgesTrue,
                           width=5,alpha=0.6,edge_color='g')


    # some math labels
    nx.draw_networkx_labels(G,pos,rename,font_size=16)
    if output is not None:
        nx.draw_networkx_edge_labels(G,pos,edgeNamesC,font_size=16,font_color='g')
        nx.draw_networkx_edge_labels(G,pos,edgeNamesI,font_size=16,font_color='r')

    plt.axis('off')
    #plt.savefig("labels_and_colors.png") # save as png
    plt.show() # display

if __name__ == "__main__":
    data = RandomMaxPairsDataset({})
    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    dataLoaderIter = iter(dataLoader)
    try:
        while True:
            display(dataLoaderIter.next())
    except StopIteration:
        print('done')
