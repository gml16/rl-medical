import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GatedGraphConv, GATConv, GENConv, GravNetConv
from torch_geometric.data import Data, Batch

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
        input = input[0].to(self.device) / 255.0
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


class Network3D_stacked_actions(nn.Module):

    def __init__(self, agents, action_history_len, number_actions, xavier=True):
        super(Network3D_stacked_actions, self).__init__()

        self.agents = agents
        self.frame_history = 1 # actions are stacked, but obs are not
        self.number_actions = number_actions
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.conv0 = nn.Conv3d(
            in_channels=self.frame_history,
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
            [nn.Linear(in_features=512+action_history_len*number_actions,
                       out_features=256).to(self.device) for _ in range(self.agents)])
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
        actions = input[1].to(self.device)
        actions = nn.functional.one_hot(actions.long(), self.number_actions).float()
        actions = actions.view(*actions.shape[:2], -1)
        obs = input[0][:, :, 0].unsqueeze(2).to(self.device) / 255.0 # takes one RoI

        output = []
        for i in range(self.agents):
            # Shared layers
            x = obs[:, i]
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
            x = torch.cat((x, actions[:, i]), 1)
            # Individual layers
            x = self.fc1[i](x)
            x = self.prelu4[i](x)
            x = self.fc2[i](x)
            x = self.prelu5[i](x)
            x = self.fc3[i](x)
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
        input1 = input[0].to(self.device) / 255.0

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


class GraphNet(nn.Module):

    def __init__(self, agents, frame_history, number_actions, xavier=True):
        super(GraphNet, self).__init__()

        self.agents = agents
        self.frame_history = frame_history
        self.number_actions = number_actions
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
        self.prelu4 = nn.PReLU().to(self.device)
        self.fc2 = nn.ModuleList(
            [nn.Linear(
                in_features=256 * 2,
                out_features=128).to(
                self.device) for _ in range(
                self.agents)])
        self.prelu5 = nn.PReLU().to(self.device)
        self.fc3 = nn.ModuleList(
            [nn.Linear(
                in_features=128 * 2,
                out_features=number_actions).to(
                self.device) for _ in range(
                self.agents)])

        self.gcn1 = GCNConv(512, 128).to(self.device)
        self.gcn2 = GCNConv(128, 16).to(self.device)

        self.fc_last = nn.Linear(
                in_features=16*agents,
                out_features=number_actions*agents).to(
                self.device)

        self.edge_index = []
        for i in range(self.agents):
            for j in range(self.agents):
                if i == j: continue
                self.edge_index.append([i, j])
        self.edge_index = torch.tensor(self.edge_index).t().contiguous().to(self.device)

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
        input1 = input[0].to(self.device) / 255.0

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
        comm = self.gcn1(input2, self.edge_index)
        comm = self.prelu4(comm)
        comm = self.gcn2(comm, self.edge_index)
        comm = self.prelu5(comm)
        comm = comm.reshape(comm.shape[0], -1) # comm is now of shape (agents, frame_history*16)
        output = self.fc_last(comm)
        return output.view(*output.shape[:-1], self.agents, self.number_actions).cpu()


class GraphNet_v2(nn.Module):

    def __init__(self, agents, frame_history, number_actions, xavier=True, graph_type="GCNConv"):
        super(GraphNet_v2, self).__init__()

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

        
        self.graph_type = graph_type
        if graph_type == "GCNConv":
            self.gcn1 = GCNConv(512, 512).to(self.device)
            self.gcn2 = GCNConv(256, 256).to(self.device)
            self.gcn3 = GCNConv(128, 128).to(self.device)
        elif graph_type == "GatedGraphConv":
            self.gcn1 = GatedGraphConv(512, 2).to(self.device)
            self.gcn2 = GatedGraphConv(256, 2).to(self.device)
            self.gcn3 = GatedGraphConv(128, 2).to(self.device)
        elif graph_type == "GATConv":
            self.gcn1 = GATConv(512, 32, heads=16, dropout=0.6).to(self.device)
            self.gcn2 = GATConv(256, 16, heads=16, dropout=0.6).to(self.device)
            self.gcn3 = GATConv(128, 16, heads=8, dropout=0.6).to(self.device)
        elif graph_type == "GENConv":
            self.gcn1 = GENConv(512, 512).to(self.device)
            self.gcn2 = GENConv(256, 256).to(self.device)
            self.gcn3 = GENConv(128, 128).to(self.device)
        elif graph_type == "GravNetConv":
            self.gcn1 = GravNetConv(512, 512, 4, 8, 2).to(self.device)
            self.gcn2 = GravNetConv(256, 256, 4, 8, 2).to(self.device)
            self.gcn3 = GravNetConv(128, 128, 4, 8, 2).to(self.device)

        self.prelu_gcn1 = nn.PReLU().to(self.device)
        self.prelu_gcn2 = nn.PReLU().to(self.device)
        self.prelu_gcn3 = nn.PReLU().to(self.device)

        self.fc_last = nn.Linear(
                in_features=16*agents,
                out_features=number_actions*agents).to(
                self.device)

        self.edge_index = []
        for i in range(self.agents):
            for j in range(self.agents):
                if i == j: continue
                self.edge_index.append([i, j])
        self.edge_index = torch.tensor(self.edge_index).t().contiguous().to(self.device)

        if xavier:
            for module in self.modules():
                if type(module) in [nn.Conv3d, nn.Linear]:
                    torch.nn.init.xavier_uniform(module.weight)

    def forward_graph(self, gcn, inputs):
        data = [Data(x=x, edge_index=self.edge_index) for x in inputs]
        batch = Batch.from_data_list(data)
        if self.graph_type == "GravNetConv":
            res = gcn(batch.x, batch)
        else:
            res = gcn(batch.x, batch.edge_index)
        return res


    def forward(self, input):
        """
        # Input is a tensor of size
        (batch_size, agents, frame_history, *image_size)
        # Output is a tensor of size
        (batch_size, agents, number_actions)
        """
        input1 = input[0].to(self.device) / 255.0

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
        comm = self.forward_graph(self.gcn1, input2)
        comm = self.prelu_gcn1(comm).view(len(input2), self.agents, 512)
        input2 = torch.cat((input2, comm), axis=-1)
        input3 = []
        for i in range(self.agents):
            x = input2[:, i]
            x = self.fc1[i](x)
            input3.append(self.prelu4[i](x))
        input3 = torch.stack(input3, dim=1)

        comm = self.forward_graph(self.gcn2, input3)
        comm = self.prelu_gcn2(comm).view(len(input3), self.agents, 256)
        input3 = torch.cat((input3, comm), axis=-1)
        input4 = []
        for i in range(self.agents):
            x = input3[:, i]
            x = self.fc2[i](x)
            input4.append(self.prelu5[i](x))
        input4 = torch.stack(input4, dim=1)

        comm = self.forward_graph(self.gcn3, input4)
        comm = self.prelu_gcn3(comm).view(len(input4), self.agents, 128)
        input4 = torch.cat((input4, comm), axis=-1)
        output = []
        for i in range(self.agents):
            x = input4[:, i]
            x = self.fc3[i](x)
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
            scheduler_step_size=100):
            type="Network3d",
            graph_type="GCNConv"):
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
                number_actions)
            self.target_network = Network3D(
                agents, frame_history, number_actions)
        elif type == "Network3d_stacked":
            self.q_network = Network3D_stacked_actions(
                agents, frame_history, number_actions)
            self.target_network = Network3D_stacked_actions(
                agents, frame_history, number_actions)
        elif type == "GraphNet":
            self.q_network = GraphNet(
                agents, frame_history, number_actions)
            self.target_network = GraphNet(
                agents, frame_history, number_actions)
        elif type == "GraphNet_v2":
            self.q_network = GraphNet_v2(
                agents, frame_history, number_actions, graph_type=graph_type)
            self.target_network = GraphNet_v2(
                agents, frame_history, number_actions, graph_type=graph_type)
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
        # States are a tuple of the stacked RoI around the agents and the previous actions
        curr_state = torch.tensor(transitions[0][0]), torch.tensor(transitions[0][1])
        next_state = torch.tensor(transitions[3][0]), torch.tensor(transitions[3][1])
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
