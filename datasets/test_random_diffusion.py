from datasets.random_diffusion import RandomDiffusionDataset
from datasets.random_diffusion import collate
import networkx as nx
from matplotlib import pyplot as plt
import torch


def display(input,output=None):
    features = input[0]
    adjM = input[1]
    gt = input[2]

    num_pairs=(features.size(1)-gt.size(1))//2
    
    G=nx.Graph()
    G.add_nodes_from(list(range(features.size(0))))

    edges = adjM._indices().t().tolist()
    G.add_edges_from(edges)

    rename={}
    senders=[]
    recievers=[]
    others=[]
    for i in range(features.size(0)):
        rename[i] = '[{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}]'.format(features[i,0],features[i,1],features[i,2],features[i,3],features[i,4],features[i,5])
        if gt[i]:
            senders.append(i)
            rename[i] = 's'+rename[i]
        else:
            others.append(i)
        if output is not None:
            rename[i]+=':{:.2}'.format(output[i,0])
        #    rename[i] = ' '
    print(output)
    #G=nx.relabel_nodes(G,rename)


    #nx.draw(G)
    #plt.show()
        
    pos=nx.spring_layout(G) # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G,pos,
                           nodelist=senders,
                           node_color='g',
                           node_size=100,
                       alpha=0.8)
    nx.draw_networkx_nodes(G,pos,
                           nodelist=others,
                           node_color='black',
                           node_size=100,
                       alpha=0.8)

    # edges
    nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
    #nx.draw_networkx_edges(G,pos,
    #                       edgelist=edges,
    #                       width=8,alpha=0.6,edge_color='graph')


    # some math labels
    nx.draw_networkx_labels(G,pos,rename,font_size=16)

    plt.axis('off')
    #plt.savefig("labels_and_colors.png") # save as png
    plt.show() # display

if __name__ == "__main__":
    data = RandomDiffusionDataset({})
    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    dataLoaderIter = iter(dataLoader)
    try:
        while True:
            display(dataLoaderIter.next())
    except StopIteration:
        print('done')
