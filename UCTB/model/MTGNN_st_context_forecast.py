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
        # pdb.set_trace()
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
    def __init__(self, 
                 gcn_true, 
                 buildA_true, 
                 gcn_depth, 
                 num_nodes, 
                 device, 
                 #------context params-------
                 context_window_size,
                 context_hidden_size,
                 context_dimensions,
                 temporal_modeling_method,
                 spatial_modeling_method,
                 context_stations,
                 #---------forecast----------
                 context_dimensions_forecast,
                 context_window_size_forecast,
                 context_stations_forecast,
                 #---------------------------
                 predefined_A=None, 
                 static_feat=None, 
                 dropout=0.3, 
                 subgraph_size=20, 
                 node_dim=40, 
                 dilation_exponential=1, 
                 conv_channels=32, 
                 residual_channels=32, 
                 skip_channels=64, 
                 end_channels=128, 
                 seq_length=12, 
                 in_dim=2, 
                 out_dim=12, 
                 layers=3, 
                 propalpha=0.05, 
                 tanhalpha=3, 
                 layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
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
        # pdb.set_trace()
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
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
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        #-------------------------------------------Context Structure-------------------------------------
        self.temporal_modeling_method = temporal_modeling_method
        self.context_hidden_size = context_hidden_size

        if temporal_modeling_method=='lstm':
            self.temporal_modeling = nn.LSTM(input_size=context_dimensions,hidden_size=context_hidden_size,num_layers=1,batch_first=True,bias=False)
            self.temporal_modeling_forecast = nn.LSTM(input_size=context_dimensions_forecast,hidden_size=context_hidden_size,num_layers=1,batch_first=True,bias=False)
        elif temporal_modeling_method=='mlp':
            self.temporal_modeling = temporalmlps(context_window_size=context_window_size,context_hidden_size=context_hidden_size,out_dim=out_dim,dimensions=context_dimensions)
            self.temporal_modeling_forecast = temporalmlps(context_window_size=context_window_size_forecast,context_hidden_size=context_hidden_size,out_dim=out_dim,dimensions=context_dimensions_forecast)
        elif temporal_modeling_method=='transformer':
            self.temporal_modeling = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=context_hidden_size,nhead=2),num_layers=1)
            self.temporal_modeling_forecast = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=context_hidden_size,nhead=2),num_layers=1)
        # input context with shape (batch_size, num_station, context_hidden_size)
        # self.stransformation = nn.Linear(in_features=context_station_num,out_features=num_nodes,bias=False)
        # output context with shape (batch_size, num_node, context_hidden_size)
        self.channel_modeling_1 = nn.Linear(in_features=context_dimensions,out_features=context_hidden_size,bias=True)
        self.channel_modeling_1_forecast = nn.Linear(in_features=context_dimensions_forecast,out_features=context_hidden_size,bias=True)
        # self.channel_modeling_2 = nn.Linear(in_features=context_hidden_size,out_features=out_dim,bias=False)
        self.spatial_modeling_method = spatial_modeling_method
        if spatial_modeling_method=='mlps':
            self.spatial_modeling = spatialmlps(context_stations=context_stations,context_hidden_size=context_hidden_size)
            self.spatial_modeling_forecast = spatialmlps(context_stations=context_stations_forecast,context_hidden_size=context_hidden_size)
        # elif spatial_modeling_method=='attn':
        #     self.spatial_modeling = 
        self.gate_linear = nn.Linear(in_features=context_hidden_size,out_features=skip_channels,bias=False)
        self.gate_linear_forecast = nn.Linear(in_features=context_hidden_size,out_features=skip_channels,bias=False)
        # self.fusion_layer = nn.Linear(in_features=2*skip_channels,out_features=skip_channels,bias=False)
        #------------------------------------------------------------------------------------------------
        self.idx = torch.arange(self.num_nodes).to(device)
    def enter_mode(self,mode):
            # crowd flow structure
            self.start_conv.requires_grad_(True if mode =='pretrain' else False)
            self.gc.requires_grad_(True if mode =='pretrain' else False)
            for filter_conv in self.filter_convs:
                filter_conv.requires_grad_(True if mode =='pretrain' else False)
            for gate_conv in self.gate_convs:
                gate_conv.requires_grad_(True if mode =='pretrain' else False)
            for residual_conv in self.residual_convs:
                residual_conv.requires_grad_(True if mode =='pretrain' else False)
            for skip_conv in self.skip_convs:
                skip_conv.requires_grad_(True if mode =='pretrain' else False)
            for gconv1 in self.gconv1:
                gconv1.requires_grad_(True if mode =='pretrain' else False)
            for gconv2 in self.gconv2:
                gconv2.requires_grad_(True if mode =='pretrain' else False)
            for norm in self.norm:
                norm.requires_grad_(True if mode =='pretrain' else False)
            # self.end_conv_1.requires_grad_(True if mode =='pretrain' else False)
            self.skip0.requires_grad_(True if mode =='pretrain' else False)
            self.skipE.requires_grad_(True if mode =='pretrain' else False)
            # context 

            self.temporal_modeling.requires_grad_(True if mode =='pretrain' else False)
            self.temporal_modeling_forecast.requires_grad_(False if mode =='pretrain' else True)
            if self.spatial_modeling_method=='mlps':
                self.spatial_modeling.requires_grad_(True if mode =='pretrain' else False)
                self.spatial_modeling_forecast.requires_grad_(False if mode =='pretrain' else True)
            self.channel_modeling_1.requires_grad_(True if mode =='pretrain' else False)
            self.channel_modeling_1_forecast.requires_grad_(False if mode =='pretrain' else True)
            self.gate_linear.requires_grad_(True if mode =='pretrain' else False)
            self.gate_linear_forecast.requires_grad_(False if mode =='pretrain' else True)
    def forward(self, input, st_context=None,st_context_forecast=None, assignment_matrix=None, assignment_matrix_forecast=None, idx=None):
        # print('in')
        # pdb.set_trace()
        seq_len = input.size(3)
        if st_context is not None:
            dimensions = st_context.size(1)
            batch_size = st_context.size(0)
            num_node = st_context.size(2)
            context_window_size = st_context.size(3)
        if st_context_forecast is not None:
            dimensions_forecast = st_context_forecast.size(1)
            num_node_forecast = st_context.size(2)
            context_window_size_forecast = st_context.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'
        # st_context with shape: (batch_size,num_node,context_window_size,num_dim)
        
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))



        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
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
        

        #------------------------Context Forward--------------------------
        #------------------Spatial Temporal Modeling----------------------
        # st_context = st_context.transpose(2,3)
        # pdb.set_trace()
        if (st_context is not None) and (st_context_forecast is not None):
            # Temporal Modeling
            # pdb.set_trace()
            if self.temporal_modeling_method=='mlp':
                context_embed = self.temporal_modeling(st_context)
                context_embed = context_embed.transpose(1,3)
                # pdb.set_trace()
                context_embed = self.channel_modeling_1(context_embed)

                context_embed_forecast = self.temporal_modeling_forecast(st_context_forecast)
                context_embed_forecast = context_embed_forecast.transpose(1,3)
                # pdb.set_trace()
                context_embed_forecast = self.channel_modeling_1_forecast(context_embed_forecast)
            elif self.temporal_modeling_method=='lstm':
                st_context = st_context.permute(0,2,3,1)
                # pdb.set_trace()
                st_context_hidenode = st_context.reshape(batch_size*num_node,context_window_size,dimensions)
                outs,_ = self.temporal_modeling(st_context_hidenode)
                context_embed = outs[:,-1:,:]
                st_context_forecast = st_context_forecast.permute(0,2,3,1)
                # pdb.set_trace()
                st_context_forecast_hidenode = st_context_forecast.reshape(batch_size*num_node_forecast,context_window_size_forecast,dimensions_forecast)
                outs_forecast,_ = self.temporal_modeling_forecast(st_context_forecast_hidenode)
                context_embed_forecast = outs_forecast[:,-1:,:]
            elif self.temporal_modeling_method=='transformer':
                # batch_size,dimension,num_node,seq_len
                st_context = st_context.permute(0,2,3,1)
                # batch_size,num_node,seq_len,dimension
                st_context_hidenode = st_context.reshape(batch_size*num_node,context_window_size,dimensions)
                # batch_size*num_node,seq_len,dimension
                st_context_hidenode = self.channel_modeling_1(st_context_hidenode)
                # batch_size*num_node,seq_len,context_hidden_size
                # pdb.set_trace()
                context_embed = self.temporal_modeling(st_context_hidenode)
                context_embed = context_embed[:,-1:,:]
                
                # batch_size,dimension,num_node,seq_len
                st_context_forecast = st_context_forecast.permute(0,2,3,1)
                # batch_size,num_node,seq_len,dimension
                st_context_forecast_hidenode = st_context_forecast.reshape(batch_size*num_node_forecast,context_window_size_forecast,dimensions_forecast)
                # batch_size*num_node,seq_len,dimension
                st_context_forecast_hidenode = self.channel_modeling_1_forecast(st_context_forecast_hidenode)
                # batch_size*num_node,seq_len,context_hidden_size
                # pdb.set_trace()
                context_embed_forecast = self.temporal_modeling_forecast(st_context_forecast_hidenode)
                context_embed_forecast = context_embed_forecast[:,-1:,:]
            # pdb.set_trace()
            # context_embed = self.channel_modeling_2(context_embed)
            
            if self.temporal_modeling_method=='lstm' or self.temporal_modeling_method=='transformer':
                context_embed = context_embed.view(batch_size,num_node,1,-1)
                context_embed = context_embed.transpose(1,2)
                context_embed_forecast = context_embed_forecast.view(batch_size,num_node_forecast,1,-1)
                context_embed_forecast = context_embed_forecast.transpose(1,2)
            # Spatial Modeling:
            if self.spatial_modeling_method=='mlps':
                context_embed = self.spatial_modeling(context_embed)
                context_embed_forecast = self.spatial_modeling_forecast(context_embed_forecast)
            # S transformation input context with shape (batch_size,1, num_station, context_hidden_size)
            
            context_embed = torch.transpose(context_embed,2,3)
            context_embed_forecast = torch.transpose(context_embed_forecast,2,3)
            # pdb.set_trace()
            if idx is not None:
                context_embed = torch.matmul(context_embed,assignment_matrix[:,idx])
                context_embed_forecast = torch.matmul(context_embed_forecast,assignment_matrix_forecast[:,idx])
            else:
                context_embed = torch.matmul(context_embed,assignment_matrix)
                context_embed_forecast = torch.matmul(context_embed_forecast,assignment_matrix_forecast)
            context_embed = torch.transpose(context_embed,2,3)
            context_embed_forecast = torch.transpose(context_embed_forecast,2,3)
            # S transformation output context with shape (batch_size, num_node, context_hidden_size)
            #------------------Fusion (Gating)------------------------------
            # pdb.set_trace()
            gate = F.relu(self.gate_linear(context_embed))
            gate_forecast = F.relu(self.gate_linear_forecast(context_embed_forecast))
            # x = torch.transpose(x,1,3)
            # x = self.fusion_layer(torch.cat((x,gate),dim=-1))
            # x = torch.transpose(x,1,3)
            gate = torch.transpose(gate,1,3)
            gate_forecast = torch.transpose(gate_forecast,1,3)
            # print(gate[0,-1,:])
            x = x + gate + gate_forecast
            # pdb.set_trace()
            #------------------------------------------------------
        if (st_context is not None):
            context_embed = []
            # Temporal Modeling
            # pdb.set_trace()
            if self.temporal_modeling_method=='mlp':
                context_embed = self.temporal_modeling(st_context)
                context_embed = context_embed.transpose(1,3)
                # pdb.set_trace()
                context_embed = self.channel_modeling_1(context_embed)
                
            elif self.temporal_modeling_method=='lstm':
                st_context = st_context.permute(0,2,3,1)
                # pdb.set_trace()
                st_context_hidenode = st_context.reshape(batch_size*num_node,context_window_size,dimensions)
                outs,_ = self.temporal_modeling(st_context_hidenode)
                context_embed = outs[:,-1:,:]
            elif self.temporal_modeling_method=='transformer':
                # batch_size,dimension,num_node,seq_len
                st_context = st_context.permute(0,2,3,1)
                # batch_size,num_node,seq_len,dimension
                st_context_hidenode = st_context.reshape(batch_size*num_node,context_window_size,dimensions)
                # batch_size*num_node,seq_len,dimension
                st_context_hidenode = self.channel_modeling_1(st_context_hidenode)
                # batch_size*num_node,seq_len,context_hidden_size
                # pdb.set_trace()
                context_embed = self.temporal_modeling(st_context_hidenode)
                context_embed = context_embed[:,-1:,:]
            # pdb.set_trace()
            # context_embed = self.channel_modeling_2(context_embed)
            
            if self.temporal_modeling_method=='lstm' or self.temporal_modeling_method=='transformer':
                context_embed = context_embed.view(batch_size,num_node,1,-1)
                context_embed = context_embed.transpose(1,2)
            # Spatial Modeling:
            if self.spatial_modeling_method=='mlps':
                context_embed = self.spatial_modeling(context_embed)
            # S transformation input context with shape (batch_size,1, num_station, context_hidden_size)
            
            context_embed = torch.transpose(context_embed,2,3)
            # pdb.set_trace()
            if idx is not None:
                context_embed = torch.matmul(context_embed,assignment_matrix[:,idx])
            else:
                context_embed = torch.matmul(context_embed,assignment_matrix)
            context_embed = torch.transpose(context_embed,2,3)
            # S transformation output context with shape (batch_size, num_node, context_hidden_size)
            #------------------Fusion (Gating)------------------------------
            # pdb.set_trace()
            gate = self.gate_linear(context_embed)
            # x = torch.transpose(x,1,3)
            # x = self.fusion_layer(torch.cat((x,gate),dim=-1))
            # x = torch.transpose(x,1,3)
            gate = torch.transpose(gate,1,3)
            x = x + gate
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x

class temporalmlps(nn.Module):
    def __init__(self,context_window_size,context_hidden_size,out_dim,dimensions,layers=3) -> None:
        super(temporalmlps,self).__init__()
        
        self.tmlps = nn.ModuleList()
        self.embedding = nn.Linear(in_features=context_window_size,out_features=context_hidden_size,bias=True)
        self.dimensions = dimensions
        for i in range(dimensions):
            self.tmlps.append(multimlps(context_hidden_size=context_hidden_size,layers=layers))
        self.output = nn.Linear(in_features=context_hidden_size,out_features=out_dim,bias=False)
        self.layers = layers
    def forward(self, input):
        # pdb.set_trace()
        x = self.embedding(input)
        x = F.relu(x)
        x_tmp = []
        for i in range(self.dimensions):
            x_tmp.append(F.relu(self.tmlps[i](x[:,i:i+1,:,:])))
        x = torch.cat(x_tmp,dim=1)
        x = self.output(x)+input[:,:,:,-1:]
        return x

class spatialmlps(nn.Module):
    def __init__(self,context_stations,context_hidden_size,layers=3) -> None:
        super(spatialmlps,self).__init__()
        
        self.smlps = nn.ModuleList()
        self.context_stations = context_stations
        for i in range(context_stations):
            self.smlps.append(multimlps(context_hidden_size=context_hidden_size,layers=layers))
        self.layers = layers
    def forward(self, input):
        # pdb.set_trace()
        x = input
        x_tmp = []
        for i in range(self.context_stations):
            x_tmp.append(F.relu(self.smlps[i](x[:,:,i:i+1,:])))
        x = torch.cat(x_tmp,dim=-2) + x
        return x

class multimlps(nn.Module):
    def __init__(self,context_hidden_size,layers=3) -> None:
        super(multimlps,self).__init__()
        
        self.mlps = nn.ModuleList()
        for i in range(layers):
            self.mlps.append(nn.Linear(in_features=context_hidden_size,out_features=context_hidden_size,bias=True))
        self.layers = layers
    def forward(self, input):
        x = input
        for i in range(self.layers):
            x = self.mlps[i](x)
            x = F.relu(x)
        return x
        
        
       