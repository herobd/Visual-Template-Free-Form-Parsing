from datasets.random_messages import RandomMessagesDataset
from datasets.random_messages import collate
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
        if features[i,0:num_pairs].any():
            id = torch.argmax(features[i,0:num_pairs])
            rename[i] = 'Sender-{}: {}'.format(id,features[i,2*num_pairs:].tolist())
            senders.append(i)
        elif features[i,num_pairs:2*num_pairs].any():
            id = torch.argmax(features[i,num_pairs:2*num_pairs])
            if output is not None:
                rename[i] = 'Reciever-{}: ['.format(id) #,output[i].tolist())
                for j in range(output.size(1)):
                    rename[i] += '{:.2f}, '.format(output[i,j])
                rename[i] += ']'
            else:
                rename[i] = 'Reciever-{}'.format(id)
            recievers.append(i)
        else:
            others.append(i)
        #    rename[i] = ' '
    #G=nx.relabel_nodes(G,rename)


    #nx.draw(G)
    #plt.show()
        
    pos=nx.spring_layout(G) # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G,pos,
                           nodelist=recievers,
                           node_color='r',
                           node_size=100,
                       alpha=0.8)
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
    data = RandomMessagesDataset({})
    dataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    dataLoaderIter = iter(dataLoader)
    try:
        while True:
            display(dataLoaderIter.next())
    except StopIteration:
        print('done')
