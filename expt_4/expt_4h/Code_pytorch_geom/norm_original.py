import torch
import torch.nn as nn

class Norm(nn.Module):

    def __init__(self, norm_type, hidden_dim,print_info=None):
        super(Norm, self).__init__()
        # assert norm_type in ['bn', 'ln', 'gn', None]
        self.norm = None
        self.print_info = print_info
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim).cuda())
            self.bias = nn.Parameter(torch.zeros(hidden_dim).cuda())

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim).cuda())

    def forward(self, graph, tensor, print_=False):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        # batch_list = graph.batch_num_nodes
        # print("Batch list",batch_list)
        # batch_size = len(batch_list)
        t,c,v=tensor.size()
        # print("Tensor device",tensor.device)
        tensor=tensor.view(t,c*v)
        batch_size=graph.num_graphs
        # print("original tensor",tensor)
        # print("Batch size", batch_size)
        # batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        # batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index=graph.batch
        # print("original batch index",batch_index)
        # print("Dimension",tensor.dim())
        __,batch_list=torch.unique(batch_index,sorted=True,return_counts=True)
        batch_list=torch.tensor(batch_list).to(tensor.device)
        # print("Batch list",batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor).to(tensor.device)
        # print("Expanded batch index",batch_index)
        mean = torch.zeros(batch_size, *tensor.shape[1:],dtype=tensor.dtype).to(tensor.device)
        # print("Mean initialized",mean)
        mean = mean.scatter_add_(0, batch_index, tensor)
        # print("mean scatter added",mean)
        mean = (mean.T / batch_list).T
        # print("Mean divided",mean)
        mean = mean.repeat_interleave(batch_list, dim=0)
        # print("Mean device",mean.device)
        # print("Self mean scale",self.mean_scale.device)
        
        # self.mean_scale=self.mean_scale.to(tensor.device)
        # self.weight=self.weight.to(tensor.device)
        # self.bias=self.bias.to(tensor.device)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:],dtype=sub.dtype).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)

        tensor=self.weight * sub / std + self.bias
        tensor=tensor.view(t,c,v)
        graph.x=tensor
        # print("X",graph.x)
        return graph
