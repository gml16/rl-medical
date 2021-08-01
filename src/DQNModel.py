import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from attention_module import AttentionModule

class Network3D(nn.Module):

    def __init__(self, agents, frame_history, number_actions, xavier=True):
        super(Network3D, self).__init__()

        self.agents = agents
        self.frame_history = frame_history
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.conv0 = nn.Conv3d(
            in_channels=frame_history,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(
            self.device)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu0 = nn.PReLU().to(self.device)
        self.conv1 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(
            self.device)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4, 4),
            padding=1).to(
            self.device)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu2 = nn.PReLU().to(self.device)
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=0).to(
            self.device)
        self.prelu3 = nn.PReLU().to(self.device)

        self.fc1 = nn.ModuleList(
            [nn.Linear(in_features=512, out_features=256).to(
                self.device) for _ in range(self.agents)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc2 = nn.ModuleList(
            [nn.Linear(in_features=256, out_features=128).to(
                self.device) for _ in range(self.agents)])
        self.prelu5 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc3 = nn.ModuleList(
            [nn.Linear(in_features=128, out_features=number_actions).to(
                self.device) for _ in range(self.agents)])

        if xavier:
            for module in self.modules():
                if type(module) in [nn.Conv3d, nn.Linear]:
                    torch.nn.init.xavier_uniform(module.weight)

    def forward(self, input):
        """
        Input is a tensor of size
        (batch_size, agents, frame_history, *image_size)
        Output is a tensor of size
        (batch_size, agents, number_actions)
        """
        input = input.to(self.device) / 255.0
        output = []
        for i in range(self.agents):
            # Shared layers
            x = input[:, i]
            x = self.conv0(x)
            x = self.prelu0(x)
            x = self.maxpool0(x)
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.prelu3(x)
            x = x.view(-1, 512)
            # Individual layers
            x = self.fc1[i](x)
            x = self.prelu4[i](x)
            x = self.fc2[i](x)
            x = self.prelu5[i](x)
            x = self.fc3[i](x)
            output.append(x)
        output = torch.stack(output, dim=1)
        return output.cpu()

class AttentionCommNet(nn.Module):

    def __init__(self,
                agents,
                frame_history,
                number_actions,
                xavier=True,
                attention=False,
                n_att_stack = 2,
                att_emb_size=128,
                n_heads=2,
                no_max_pool=False):
        super(AttentionCommNet, self).__init__()

        self.agents = agents
        self.frame_history = frame_history
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.no_max_pool = no_max_pool

        self.conv0 = nn.Conv3d(
            in_channels=frame_history,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(
            self.device)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu0 = nn.PReLU().to(self.device)
        self.conv1 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(
            self.device)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4, 4),
            padding=1).to(
            self.device)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu2 = nn.PReLU().to(self.device)
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=0).to(
            self.device)
        self.prelu3 = nn.PReLU().to(self.device)

        conv3w = 2
        conv3h = 2
        conv3z = 2

        '''
        # create x,y,z coordinate matrices to append to convolution output
        xmap = np.linspace(-np.ones(conv2h), np.ones(conv2h), num=conv2w, endpoint=True, axis=0)
        xmap = torch.tensor(np.expand_dims(np.expand_dims(xmap,0),0), dtype=torch.float32, requires_grad=False)
        ymap = np.linspace(-np.ones(conv2w), np.ones(conv2w), num=conv2h, endpoint=True, axis=1)
        ymap = torch.tensor(np.expand_dims(np.expand_dims(ymap,0),0), dtype=torch.float32, requires_grad=False)
        self.register_buffer("xymap", torch.cat((xmap,ymap),dim=1)) # shape (1, 2, conv2w, conv2h)
        '''

        x = np.linspace(-1,1,conv3w)
        y = np.linspace(-1,1,conv3h)
        z = np.linspace(-1,1,conv3z)

        xv, yv, zv = np.meshgrid(x, y, z, sparse=False, indexing='ij')

        xv= torch.tensor(np.expand_dims(np.expand_dims(xv,0),0), dtype=torch.float32, requires_grad=False)
        yv = torch.tensor(np.expand_dims(np.expand_dims(yv,0),0), dtype=torch.float32, requires_grad=False)
        zv = torch.tensor(np.expand_dims(np.expand_dims(zv,0),0), dtype=torch.float32, requires_grad=False)

        self.register_buffer("xyzmap", torch.cat((xv,yv,zv),dim=1)) # shape (1, 3, conv3w, conv3h, conv3h)

        # an "attendable" entity has 64 CNN channels + 3 coordinate channels = 67 features
        self.att_elem_size = 64 + 3
        # create attention module with n_heads heads and remember how many times to stack it
        self.n_att_stack = n_att_stack #how many times the attentional module is to be stacked (weight-sharing -> reuse)
        self.attMod = AttentionModule(conv3w*conv3h*conv3z, self.att_elem_size, att_emb_size, n_heads)

        if not self.no_max_pool:
            self.fc1 = nn.ModuleList(
                [nn.Linear(
                    in_features=self.att_elem_size * 2,
                    out_features=256).to(
                    self.device) for _ in range(
                    self.agents)])
        else:
            self.fc1 = nn.ModuleList(
                [nn.Linear(
                    in_features=self.att_elem_size * conv3w * conv3h * conv3z * 2,
                    out_features=256).to(
                    self.device) for _ in range(
                    self.agents)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc2 = nn.ModuleList(
            [nn.Linear(
                in_features=256 * 2,
                out_features=128).to(
                self.device) for _ in range(
                self.agents)])
        self.prelu5 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc3 = nn.ModuleList(
            [nn.Linear(
                in_features=128 * 2,
                out_features=number_actions).to(
                self.device) for _ in range(
                self.agents)])

        self.attention = attention
        if self.attention:
                self.comm_att1 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])
                self.comm_att2 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])
                self.comm_att3 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])

        if xavier:
            for module in self.modules():
                if type(module) in [nn.Conv3d, nn.Linear]:
                    torch.nn.init.xavier_uniform(module.weight)

    def forward(self, input):
        """
        # Input is a tensor of size
        (batch_size, agents, frame_history, *image_size)
        # Output is a tensor of size
        (batch_size, agents, number_actions)
        """
        input1 = input.to(self.device) / 255.0

        # Shared layers
        input2 = []
        for i in range(self.agents):
            x = input1[:, i]
            x = self.conv0(x)
            x = self.prelu0(x)
            x = self.maxpool0(x)
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.prelu3(x)
            #Attention mechanism
            batchsize = x.size(0)
            batch_maps = self.xyzmap.repeat(batchsize,1,1,1,1,)
            x = torch.cat((x,batch_maps),1)
            x = x.view(x.size(0),x.size(1), -1).transpose(1,2)
            for i_att in range(self.n_att_stack):
                x = self.attMod(x)
            if not self.no_max_pool:
                kernelsize = x.shape[1]
                if type(kernelsize) == torch.Tensor:
                    kernelsize = kernelsize.item()
                x = F.max_pool1d(x.transpose(1,2), kernel_size=kernelsize)
                x = x.view(-1, self.att_elem_size)
            else:
                x = x.view(x.shape[0],-1)
            input2.append(x)
        input2 = torch.stack(input2, dim=1)

        # Communication layers
        if self.attention:
            comm = torch.cat([torch.sum((input2.transpose(1, 2) * nn.Softmax(dim=0)(self.comm_att1[i])), axis=2).unsqueeze(0)
                              for i in range(self.agents)])

        else:
            comm = torch.mean(input2, axis=1)
            comm = comm.unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        input3 = []
        for i in range(self.agents):
            x = input2[:, i]
            x = self.fc1[i](torch.cat((x, comm[i]), axis=-1))
            input3.append(self.prelu4[i](x))
        input3 = torch.stack(input3, dim=1)

        if self.attention:
            comm = torch.cat([torch.sum((input3.transpose(1, 2) * nn.Softmax(dim=0)(self.comm_att2[i])), axis=2).unsqueeze(0)
                              for i in range(self.agents)])
        else:
            comm = torch.mean(input3, axis=1)
            comm = comm.unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        input4 = []
        for i in range(self.agents):
            x = input3[:, i]
            x = self.fc2[i](torch.cat((x, comm[i]), axis=-1))
            input4.append(self.prelu5[i](x))
        input4 = torch.stack(input4, dim=1)

        if self.attention:
            comm = torch.cat([torch.sum((input4.transpose(1, 2) * nn.Softmax(dim=0)(self.comm_att3[i])), axis=2).unsqueeze(0)
                              for i in range(self.agents)])
        else:
            comm = torch.mean(input4, axis=1)
            comm = comm.unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        output = []
        for i in range(self.agents):
            x = input4[:, i]
            x = self.fc3[i](torch.cat((x, comm[i]), axis=-1))
            output.append(x)
        output = torch.stack(output, dim=1)

        return output.cpu()

class CommAttentionNet(nn.Module):

    def __init__(self,
                agents,
                frame_history,
                number_actions,
                xavier=True,
                attention=False,
                n_att_stack = 2,
                att_emb_size=512,
                n_heads=2,
                no_max_pool=False):
        super(CommAttentionNet, self).__init__()

        self.agents = agents
        self.frame_history = frame_history
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.no_max_pool = no_max_pool

        self.conv0 = nn.Conv3d(
            in_channels=frame_history,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(
            self.device)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu0 = nn.PReLU().to(self.device)
        self.conv1 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(
            self.device)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4, 4),
            padding=1).to(
            self.device)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu2 = nn.PReLU().to(self.device)
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=0).to(
            self.device)
        self.prelu3 = nn.PReLU().to(self.device)

        self.n_att_stack = n_att_stack #how many times the attentional module is to be stacked (weight-sharing -> reuse)


        conv3w = 2
        conv3h = 2
        conv3z = 2

        #Initially an "attendable" agent has 64 CNN channels with feature maps of size 2*2*2 = 64 * 8 = 512 features
        self.att_elem_size_1 = 64 * conv3h * conv3w * conv3z
        self.attMod1 = AttentionModule(agents, self.att_elem_size_1, att_emb_size, n_heads)

        if not self.no_max_pool:
            self.fc1 = nn.ModuleList(
                [nn.Linear(
                    in_features=self.att_elem_size_1 * 2,
                    out_features=256).to(
                    self.device) for _ in range(
                    self.agents)])
        else:
            self.fc1 = nn.ModuleList(
                [nn.Linear(
                    in_features=self.att_elem_size_1 * agents * 2,
                    out_features=256).to(
                    self.device) for _ in range(
                    self.agents)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])

        #After the first fc layer an "attendable" agent has 256 features
        self.att_elem_size_2 = 256
        self.attMod2 = AttentionModule(agents, self.att_elem_size_2, att_emb_size/2, n_heads)

        if not self.no_max_pool:
            self.fc2 = nn.ModuleList(
                [nn.Linear(
                    in_features=256 * 2,
                    out_features=128).to(
                    self.device) for _ in range(
                    self.agents)])
        else:
            self.fc2 = nn.ModuleList(
                [nn.Linear(
                    in_features=256 * agents * 2,
                    out_features=128).to(
                    self.device) for _ in range(
                    self.agents)])
        self.prelu5 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])

        #After the second fc layer an "attendable" agent has 128 features
        self.att_elem_size_3 = 128
        self.attMod3 = AttentionModule(agents, self.att_elem_size_3, att_emb_size/4, n_heads)

        if not self.no_max_pool:
            self.fc3 = nn.ModuleList(
                [nn.Linear(
                    in_features=128 * 2,
                    out_features=number_actions).to(
                    self.device) for _ in range(
                    self.agents)])
        else:
            self.fc3 = nn.ModuleList(
                [nn.Linear(
                    in_features=128 * agents * 2,
                    out_features=number_actions).to(
                    self.device) for _ in range(
                    self.agents)])

        self.attention = attention
        if self.attention:
                self.comm_att1 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])
                self.comm_att2 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])
                self.comm_att3 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])

        if xavier:
            for module in self.modules():
                if type(module) in [nn.Conv3d, nn.Linear]:
                    torch.nn.init.xavier_uniform(module.weight)

    def forward(self, input):
        """
        # Input is a tensor of size
        (batch_size, agents, frame_history, *image_size)
        # Output is a tensor of size
        (batch_size, agents, number_actions)
        """
        input1 = input.to(self.device) / 255.0

        # Shared layers
        input2 = []
        for i in range(self.agents):
            x = input1[:, i]
            x = self.conv0(x)
            x = self.prelu0(x)
            x = self.maxpool0(x)
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.prelu3(x)
            x = x.view(-1, 512)
            input2.append(x)
        input2 = torch.stack(input2, dim=1)

        comm = self.attMod1(input2)
        for i_att in range(self.n_att_stack-1):
            comm = self.attMod1(comm)

        if not self.no_max_pool:
            kernelsize = comm.shape[1]
            if type(kernelsize) == torch.Tensor:
                kernelsize = kernelsize.item()
            comm = F.max_pool1d(comm.transpose(1,2), kernel_size=kernelsize)
            comm = comm.view(-1, self.att_elem_size_1).unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        else:
            comm = comm.view(comm.shape[0], self.att_elem_size_1, self.agents).permute(2,0,1)

        input3 = []
        for i in range(self.agents):
            x = input2[:, i]
            x = self.fc1[i](torch.cat((x, comm[i]), axis=-1))
            input3.append(self.prelu4[i](x))
        input3 = torch.stack(input3, dim=1)

        comm = self.attMod2(input3)
        for i_att in range(self.n_att_stack-1):
            comm = self.attMod2(comm)

        if not self.no_max_pool:
            kernelsize = comm.shape[1]
            if type(kernelsize) == torch.Tensor:
                kernelsize = kernelsize.item()
            comm = F.max_pool1d(comm.transpose(1,2), kernel_size=kernelsize)
            comm = comm.view(-1, self.att_elem_size_2).unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        else:
            comm = comm.view(comm.shape[0], self.att_elem_size_2, self.agents).permute(2,0,1)

        input4 = []
        for i in range(self.agents):
            x = input3[:, i]
            x = self.fc2[i](torch.cat((x, comm[i]), axis=-1))
            input4.append(self.prelu5[i](x))
        input4 = torch.stack(input4, dim=1)

        comm = self.attMod3(input4)
        for i_att in range(self.n_att_stack-1):
            comm = self.attMod3(comm)
            
        if not self.no_max_pool:
            kernelsize = comm.shape[1]
            if type(kernelsize) == torch.Tensor:
                kernelsize = kernelsize.item()
            comm = F.max_pool1d(comm.transpose(1,2), kernel_size=kernelsize)
            comm = comm.view(-1, self.att_elem_size_3).unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        else:
            comm = comm.view(comm.shape[0], self.att_elem_size_3, self.agents).permute(2,0,1)

        output = []
        for i in range(self.agents):
            x = input4[:, i]
            x = self.fc3[i](torch.cat((x, comm[i]), axis=-1))
            output.append(x)
        output = torch.stack(output, dim=1)

        return output.cpu()

class CommNet(nn.Module):

    def __init__(self, agents, frame_history, number_actions, xavier=True, attention=False):
        super(CommNet, self).__init__()

        self.agents = agents
        self.frame_history = frame_history
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.conv0 = nn.Conv3d(
            in_channels=frame_history,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(
            self.device)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu0 = nn.PReLU().to(self.device)
        self.conv1 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(
            self.device)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4, 4),
            padding=1).to(
            self.device)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu2 = nn.PReLU().to(self.device)
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=0).to(
            self.device)
        self.prelu3 = nn.PReLU().to(self.device)

        self.fc1 = nn.ModuleList(
            [nn.Linear(
                in_features=512 * 2,
                out_features=256).to(
                self.device) for _ in range(
                self.agents)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc2 = nn.ModuleList(
            [nn.Linear(
                in_features=256 * 2,
                out_features=128).to(
                self.device) for _ in range(
                self.agents)])
        self.prelu5 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc3 = nn.ModuleList(
            [nn.Linear(
                in_features=128 * 2,
                out_features=number_actions).to(
                self.device) for _ in range(
                self.agents)])

        self.attention = attention
        if self.attention:
                self.comm_att1 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])
                self.comm_att2 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])
                self.comm_att3 = nn.ParameterList([nn.Parameter(torch.randn(agents)) for _ in range(agents)])

        if xavier:
            for module in self.modules():
                if type(module) in [nn.Conv3d, nn.Linear]:
                    torch.nn.init.xavier_uniform(module.weight)

    def forward(self, input):
        """
        # Input is a tensor of size
        (batch_size, agents, frame_history, *image_size)
        # Output is a tensor of size
        (batch_size, agents, number_actions)
        """
        input1 = input.to(self.device) / 255.0

        # Shared layers
        input2 = []
        for i in range(self.agents):
            x = input1[:, i]
            x = self.conv0(x)
            x = self.prelu0(x)
            x = self.maxpool0(x)
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.prelu3(x)
            x = x.view(-1, 512)
            input2.append(x)
        input2 = torch.stack(input2, dim=1)

        # Communication layers
        if self.attention:
            comm = torch.cat([torch.sum((input2.transpose(1, 2) * nn.Softmax(dim=0)(self.comm_att1[i])), axis=2).unsqueeze(0)
                              for i in range(self.agents)])

        else:
            comm = torch.mean(input2, axis=1)
            comm = comm.unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        input3 = []
        for i in range(self.agents):
            x = input2[:, i]
            x = self.fc1[i](torch.cat((x, comm[i]), axis=-1))
            input3.append(self.prelu4[i](x))
        input3 = torch.stack(input3, dim=1)

        if self.attention:
            comm = torch.cat([torch.sum((input3.transpose(1, 2) * nn.Softmax(dim=0)(self.comm_att2[i])), axis=2).unsqueeze(0)
                              for i in range(self.agents)])
        else:
            comm = torch.mean(input3, axis=1)
            comm = comm.unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        input4 = []
        for i in range(self.agents):
            x = input3[:, i]
            x = self.fc2[i](torch.cat((x, comm[i]), axis=-1))
            input4.append(self.prelu5[i](x))
        input4 = torch.stack(input4, dim=1)

        if self.attention:
            comm = torch.cat([torch.sum((input4.transpose(1, 2) * nn.Softmax(dim=0)(self.comm_att3[i])), axis=2).unsqueeze(0)
                              for i in range(self.agents)])
        else:
            comm = torch.mean(input4, axis=1)
            comm = comm.unsqueeze(0).repeat(self.agents, *[1]*len(comm.shape))
        output = []
        for i in range(self.agents):
            x = input4[:, i]
            x = self.fc3[i](torch.cat((x, comm[i]), axis=-1))
            output.append(x)
        output = torch.stack(output, dim=1)

        return output.cpu()


class DQN:
    # The class initialisation function.
    def __init__(
            self,
            agents,
            frame_history,
            logger,
            number_actions=6,
            type="Network3d",
            collective_rewards=False,
            attention=False,
            lr=1e-3,
            scheduler_gamma=0.9,
            scheduler_step_size=100,
            no_max_pool=False):
        self.agents = agents
        self.number_actions = number_actions
        self.frame_history = frame_history
        self.logger = logger
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.log(f"Using {self.device}")
        # Create a Q-network, which predicts the q-value for a particular state
        if type == "Network3d":
            self.q_network = Network3D(
                agents,
                frame_history,
                number_actions).to(
                self.device)
            self.target_network = Network3D(
                agents, frame_history, number_actions).to(
                self.device)
        elif type == "CommNet":
            self.q_network = CommNet(
                agents,
                frame_history,
                number_actions,
                attention=attention).to(
                self.device)
            self.target_network = CommNet(
                agents,
                frame_history,
                number_actions,
                attention=attention).to(
                self.device)
        elif type == "AttentionCommNet":
            self.q_network = AttentionCommNet(
                agents,
                frame_history,
                number_actions,
                attention=attention,
                no_max_pool=no_max_pool).to(
                self.device)
            self.target_network = AttentionCommNet(
                agents,
                frame_history,
                number_actions,
                attention=attention,
                no_max_pool=no_max_pool).to(
                self.device)
        elif type == "CommAttentionNet":
            self.q_network = CommAttentionNet(
                agents,
                frame_history,
                number_actions,
                attention=attention,
                no_max_pool=no_max_pool).to(
                self.device)
            self.target_network = CommAttentionNet(
                agents,
                frame_history,
                number_actions,
                attention=attention,
                no_max_pool=no_max_pool).to(
                self.device)
        if collective_rewards == "attention":
            self.q_network.rew_att = nn.Parameter(torch.randn(agents, agents))
            self.target_network.rew_att = nn.Parameter(torch.randn(agents, agents))
        self.copy_to_target_network()
        # Freezes target network
        self.target_network.train(False)
        for p in self.target_network.parameters():
            p.requires_grad = False
        # Define the optimiser which is used when updating the Q-network. The
        # learning rate determines how big each gradient step is during
        # backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimiser, step_size=scheduler_step_size, gamma=scheduler_gamma)
        self.collective_rewards = collective_rewards

    def copy_to_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, name="dqn.pt", forced=False):
        self.logger.save_model(self.q_network.state_dict(), name, forced)

    # Function that is called whenever we want to train the Q-network. Each
    # call to this function takes in a transition tuple containing the data we
    # use to update the Q-network.
    def train_q_network(self, transitions, discount_factor):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transitions, discount_factor)
        # Compute the gradients based on this loss, i.e. the gradients of the
        # loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, discount_factor):
        '''
        Transitions are tuple of shape
        (states, actions, rewards, next_states, dones)
        '''
        curr_state = torch.tensor(transitions[0])
        next_state = torch.tensor(transitions[3])
        terminal = torch.tensor(transitions[4]).type(torch.int)

        rewards = torch.clamp(
            torch.tensor(
                transitions[2], dtype=torch.float32), -1, 1)
        # Collective rewards here refers to adding the (potentially weighted) average reward of all agents
        if self.collective_rewards == "mean":
            rewards += torch.mean(rewards, axis=1).unsqueeze(1).repeat(1, rewards.shape[1])
        elif self.collective_rewards == "attention":
            rewards = rewards + torch.matmul(rewards, nn.Softmax(dim=0)(self.q_network.rew_att))

        y = self.target_network.forward(next_state)
        # dim (batch_size, agents, number_actions)
        y = y.view(-1, self.agents, self.number_actions)
        # Get the maximum prediction for the next state from the target network
        max_target_net = y.max(-1)[0]

        # dim (batch_size, agents, number_actions)
        network_prediction = self.q_network.forward(curr_state).view(
            -1, self.agents, self.number_actions)
        isNotOver = (torch.ones(*terminal.shape) - terminal)
        # Bellman equation
        batch_labels_tensor = rewards + isNotOver * \
            (discount_factor * max_target_net.detach())

        actions = torch.tensor(transitions[1], dtype=torch.long).unsqueeze(-1)
        y_pred = torch.gather(network_prediction, -1, actions).squeeze()

        return torch.nn.SmoothL1Loss()(batch_labels_tensor.flatten(), y_pred.flatten())
