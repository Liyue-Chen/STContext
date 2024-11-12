



from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
import pdb

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho

class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)


    def forward(self,x):
        #adj = adj + torch.eye(adj.size(0)).to(x.device)
        #d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)


        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2



class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj



class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True,poidim=35,poioutdim=10,graph_AM=None,IIFdata=None,S_method=None,T_method=None,gcnlayer=2,hiddendim=10,timevec=1,interaction='none'):
        super(gtnet, self).__init__()
        import numpy as np
        import random
        # seed=21
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # np.random.seed(seed)
        # random.seed(seed)
        # torch.backends.cudnn.deterministic = True

        # torch.backends.cudnn.benchmark = False
        
        self.interaction=interaction
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.poidim=poidim
        self.poioutdim=poioutdim
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)


        
        if timevec==8:
            self.timevec=15
        else:
            self.timevec=int(7+24*timevec)

        if timevec==16:
            self.timevec=23
            

        self.hidden_gcn=48
        self.Graph=graph_AM
        
        # hidden_gcn=hidden_mlps1
        hidden_timev=int((self.timevec+10)*2/3)
        hidden_mlppoi=int((self.poioutdim+self.poioutdim)*2/3)
        hidden_metalearn=int((self.timevec+64*10)*2/3)
        outdim_mlp=74
        if S_method!='none' and T_method=='none':
            outdim_mlp=64
            
                
        elif S_method!='none' and T_method!='none':
            outdim_mlp=64
            if self.interaction=='add' or self.interaction=='gating' or self.interaction=='concat':
                outdim_mlp=10
        
        if T_method=='meta-learner':
            outdim_mlp=64
        
        
        hidden_mlps1=int((self.poidim+outdim_mlp)*2/3)
        
        
        self.mlp_S_1_=MLP(self.poidim,hidden_mlps1,outdim_mlp)
        self.device=device
        hidden_dims=[]
        for i in range(gcnlayer):
            hidden_dims.append(self.hidden_gcn)
            
        
        self.gcn_s3_=GCN(self.poidim,64,hidden_dims)
        
        self.timev_mlp_=MLP(self.timevec,hidden_timev,10)
        
        self.timev_metalearn_=MLP(self.timevec,hidden_metalearn,64*10)
        self.pt_trans_=PT_trans(self.poioutdim,self.timevec)
        
        
        hid_mlpW=int((self.poioutdim+10)*2/3)
        self.mlp_w=MLP(10,10,10)
        self.mlppoi_=MLP(self.poioutdim,hidden_mlppoi,self.poioutdim)
        self.S_method=S_method
        self.T_method=T_method

            
        self.IIF=torch.Tensor(IIFdata).to(device)
        
        if self.T_method=="meta-learner":
            self.end_conv_fuse=skip_channels+self.poioutdim
        elif  self.T_method=="time-concat":
            self.end_conv_fuse=skip_channels+10
            if self.interaction=='concat':
                self.end_conv_fuse=skip_channels+20
        elif  self.T_method=="deepstn":
            self.end_conv_fuse=skip_channels+self.poioutdim   
        else:
            self.end_conv_fuse=skip_channels
        

        
       
        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1_sgcn = nn.Conv2d(in_channels=self.end_conv_fuse,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2_sgcn = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)


        self.idx = torch.arange(self.num_nodes).to(device)
        self.fusion_layer_1 = CrossTransformer(source_num_frames = self.num_nodes,
                                tgt_num_frames = self.num_nodes,
                                dim=64,
                                depth=2,
                                heads=4,
                                mlp_dim=256,
                                dropout=0.5,
                                emb_dropout=0.1
                                )

        self.fusion_layer_2 = CrossTransformer(source_num_frames = self.num_nodes,
                                tgt_num_frames = self.num_nodes,
                                dim=poioutdim,
                                depth=2,
                                heads=2,
                                mlp_dim=64,
                                dropout=0.5,
                                emb_dropout=0.1
                                )
        
        self.mlp_final=MLP(64,256,64)


    def forward(self, input, idx=None):
        
        poiindex=-self.poidim
        Time_V=input[:,:,:,-(self.poidim+self.timevec):poiindex]
        context=input[:,:,:,poiindex:]
        insert_=input[:,:,:,:-(self.poidim+self.timevec)]
        seq_len = insert_.size(3)
        # pdb.set_trace()
        assert seq_len==self.seq_length, 'insert_ sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            insert_ = nn.functional.pad(insert_,(self.receptive_field-self.seq_length,0,0,0))



        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(insert_)
        skip = self.skip0(F.dropout(insert_, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        
        
        
        if self.S_method=="mlp":
            
            context=context.squeeze()
            context=self.mlp_S_1_(context)

        elif self.S_method=="att1_self":
            
            context=context.squeeze()
            context=self.mlp_S_1_(context)
            context=self.fusion_layer_1(context,context)

        elif self.S_method=="att2_cross":
            
            context=context.squeeze()
            context=self.mlp_S_1_(context)
            context_old=context
            x_=x.squeeze()
            x_=x_.permute(0,2,1)
            context=self.fusion_layer_1(context,x_)
            context=context+context_old

        elif self.S_method=="gcn":
            context=context.squeeze()
            context=self.gcn_s3_(self.Graph,context,self.device)

            
        elif self.S_method=="IIF":
            context=context.squeeze()
            context=self.mlp_S_1_(context)

            
            
        if self.T_method=="meta-learner":
            Time_V=Time_V.squeeze()
            Time_V=Time_V[:,0,:]
            Time_V=self.timev_metalearn_(Time_V)
            Time_V=Time_V.view(Time_V.size(0),64,10)

            context=torch.matmul(context, Time_V)

            
        elif self.T_method=="time-concat":
            Time_V=Time_V.squeeze()
            Time_V=self.timev_mlp_(Time_V)

        elif self.T_method=="deepstn":  
            Time_V=Time_V.squeeze()
            Time_V=Time_V.permute(0,2,1)
            input2=context
            time_in=torch.mean(Time_V,dim=2)
            P_N = self.poioutdim
            #P_N代表获取poi种类个数。
            context = self.pt_trans_(input2, time_in, P_N, self.poioutdim, input2.shape[1],'s-constant')
            #poi_time_代表poi特征得到的representation。
            context=self.mlppoi_(context)
        else:
            context=context

        
        if self.T_method=="meta-learner":
            context=context.permute(0, 2, 1)
            context=context.unsqueeze(3)
            x=torch.cat((x,context),dim=1)

        else:
 
            if self.S_method!='none':
        
                if self.interaction=='none': 
                    context=context.permute(0, 2, 1)
                    context=context.unsqueeze(3)
                    x=x+context
        
            if self.T_method!='none':
                
                if self.interaction=='gating':
                    Time_V=torch.mul(Time_V,context)
                    
                if self.interaction=='add':
                    Time_V=Time_V+context
                
                if self.interaction=='concat':
                    Time_V=torch.cat((Time_V,context),dim=2)

                
                Time_V=Time_V.permute(0, 2, 1)
                Time_V=Time_V.unsqueeze(3)
                x=torch.cat((x,Time_V),dim=1)


            
        
        x = F.relu(self.end_conv_1_sgcn(x))
        x = self.end_conv_2_sgcn(x)
        
        
        
        return x






import torch
import torch.nn as nn

import math
import torch.nn.functional as F





import torch
import torch.nn as nn
import torch.nn.functional as F




class GCN(nn.Module):  # GCN模型，向空域的第一个图卷积
    def __init__(self, in_c, out_c,hidden_dims):
        super(GCN, self).__init__()  # 表示继承父类的所有属性和方法
        self.layers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(in_c, hidden_dim))
            in_c = hidden_dim
            
        self.layers.append(nn.Linear(hidden_dims[-1], out_c))
        self.act = nn.ReLU()  # 定义激活函数

    def forward(self, graph,flow,device):
        graph_data = torch.Tensor(graph).to(device)  # [N, N] 邻接矩阵，并且将数据送入设备
        
 
        graph_data = GCN.process_graph(graph_data,device)  # 变换邻接矩阵 \hat A = D_{-1/2}*A*D_{-1/2}

        flow_x = flow # [B, N, D]  流量数据

        B, N = flow_x.size(0), flow_x.size(1)  # batch_size、节点数

        # flow_x = flow_x.view(B, N, -1)  # [B, N, H*D] H = 6, D = 1把最后两维缩减到一起了，这个就是把历史时间的特征放一起
        
        x=flow_x
        for layer in self.layers:
            x = self.act((torch.matmul(graph_data, layer(x))))



        return x  # 第２维的维度扩张


    @staticmethod
    def process_graph(graph_data,device):  # 这个就是在原始的邻接矩阵之上，再次变换，也就是\hat A = D_{-1/2}*A*D_{-1/2}
      
        
        N = graph_data.size(0) # 获得节点的个数
        matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)  # 定义[N, N]的单位矩阵
        graph_data += matrix_i  # [N, N]  ,就是 A+I

        degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
        degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
        degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0

        degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵

        return torch.mm(degree_matrix, graph_data)  # 返回 \hat A=D^(-1) * A ,这个等价于\hat A = D_{-1/2}*A*D_{-1/2}
    
    
    
class MLP(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(MLP, self).__init__()
    
        self.linear1 = nn.Linear(n_i, n_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_h, n_o)
 
    def forward(self, input):
        return self.linear2(self.relu(self.linear1(input)))






class T_trans(nn.Module):
    def __init__(self, inputs_shape,T1,T2):
        super(T_trans,self).__init__()
        
        
        T_=inputs_shape
        self.relu = torch.nn.ReLU().to('cuda:0')
        self.Linnear1=torch.nn.Linear(int(T_),int(T1)).to('cuda:0')
        self.Linnear2=torch.nn.Linear(int(T1),int(T2)).to('cuda:0')
    # forward 定义前向传播
    def forward(self, x):
        x = x.to('cuda:0')
        
        x=self.Linnear1(x)
        x =self.relu(x)
        x=self.Linnear2(x)
        x =self.relu(x)     
        return x
    
    

class PT_trans(nn.Module):
    def __init__(self,P_N,timevec):
        super(PT_trans,self).__init__()
        self.netlist=[]
        
        self.timevec=timevec
        for i in range(P_N):
            self.netlist.append(T_trans(self.timevec,10,1))
            
    def forward(self,poi_in, time_in,P_N, T_F, Num_node,way):
        
        net1= T_trans(time_in.shape[-1],Num_node/2,Num_node)
        # net2= T_trans(time_in.shape[-1],T_F,1)
    
        #for循环就是说，Poi有几个，我们会创建多少个全连接网络去学习这一类poi的影响，至于是这一类poi对全局的影响还是局部的影响我们使用if else划分开来。
        for i in range (P_N):
            #假如是选择time_vary-space_vary_non_graph方法
            if(way=='s-vary'):
                T_m=net1(time_in)
                # print(T_m.shape)

                # T_m=T_trans(time_in, Num_node/2,Num_node)
                T_m=T_m.unsqueeze(1)
                ll = T_m if i == 0 else torch.cat((ll,T_m),1)  
            #假如是选择time_vary-space_constant_non方法
            elif (way=='s-constant'):
                T_m=self.netlist[i](time_in)
                if i==0:
                    ll=T_m
                else:
                    ll=torch.cat((ll,T_m),1) 
                    
                    
        T_x0 = ll
        
        #space-vary学习的影响权重需要转置一下才可以与输入poi分布对应。    
        if (way == 's-vary'):
            # print("l",T_x0.shape)
            T_x0=T_x0.permute(0,2,1)
        else:
            #s-constant学习的影响权重需要扩充一个维度。同时由于全局地区该类poi影响是相同的，因此直接复制一下，从而才可与输入poi维度对应。
            T_x0 = T_x0.unsqueeze(1).repeat(1,Num_node,1)  
        # print(T_x0.shape,poi_in.shape)
        poi_time = torch.mul(T_x0, poi_in)
        return poi_time





class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        source_x = source_x + self.pos_embedding_s[:, : n_s]
        target_x = target_x + self.pos_embedding_t[:, : n_t]
        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t
    
    

class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x = attn(target_x, source_x, source_x) + target_x
            target_x = ff(target_x) + target_x
        return target_x



class PreNorm_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)
    
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
 
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        # print("1", q.shape, k.shape, v.shape)
        b, n, _, h = *q.shape, self.heads
        # qkv = self.to_qkv(x).chunk(3, dim = -1)

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        # print("2", q.shape, k.shape, v.shape)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


