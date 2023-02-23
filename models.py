import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.utils.weight_norm as weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """
    Simple multi-layer neural network.

    Example:
    >>> torch.manual_seed(123)
    >>> net = stribor.net.MLP(2, [64, 64], 1)
    >>> net(torch.randn(1, 2))
    tensor([[-0.0132]], grad_fn=<AddmmBackward>)

    Args:
        in_dim (int): Input size
        hidden_dims (List[int]): Hidden dimensions
        out_dim (int): Output size
        activation (str, optional): Activation function from `torch.nn`.
            Default: 'Tanh'
        final_activation (str, optional): Last activation. Default: None
        wrapper_func (callable, optional): Wrapper function for `nn.Linear`,
            e.g. st.util.spectral_norm. Default: None
    """

    def __init__(
        self,
        in_dim,
        hidden_dims,
        out_dim,
        activation="Tanh",
        final_activation=None,
        wrapper_func=None,
        **kwargs,
    ):
        super().__init__()

        if not wrapper_func:
            wrapper_func = lambda x: x

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)
        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(getattr(nn, activation)())
            layers.append(wrapper_func(nn.Linear(hidden_dims[i], hidden_dims[i + 1])))
        layers[-1].bias.data.fill_(0.0)

        if final_activation is not None:
            layers.append(getattr(nn, final_activation)())

        self.net = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        """
        Args:
            x (tensor): Input with shape (..., in_dim)

        Returns:
            y (tensor): Output with shape (..., out_dim)
        """

        return self.net(x)


######################################
### PERMUTATION EQUIVARIANT LAYER  ###
######################################


# equivariant layer with global concat & residual connections inside this module  & weight_norm
# ordered: first update global, then local
class EPiC_layer(nn.Module):
    def __init__(self, local_in_dim, hid_dim, latent_dim):
        super(EPiC_layer, self).__init__()
        self.fc_global1 = weight_norm(nn.Linear(int(2 * hid_dim) + latent_dim, hid_dim))
        self.fc_global2 = weight_norm(nn.Linear(hid_dim, latent_dim))
        self.fc_local1 = weight_norm(nn.Linear(local_in_dim + latent_dim, hid_dim))
        self.fc_local2 = weight_norm(nn.Linear(hid_dim, hid_dim))

    def forward(
        self, x_global, x_local
    ):  # shapes: x_global[b,latent], x_local[b,n,latent_local]
        batch_size, n_points, latent_local = x_local.size()
        latent_global = x_global.size(1)

        ### meansum pooling
        x_pooled_mean = x_local.mean(1, keepdim=False)
        x_pooled_sum = x_local.sum(1, keepdim=False)
        x_pooledCATglobal = torch.cat(
            [x_pooled_mean, x_pooled_sum, x_global], 1
        )  # meansum pooling

        ### phi global
        x_global1 = F.leaky_relu(
            self.fc_global1(x_pooledCATglobal)
        )  # new intermediate step
        x_global = F.leaky_relu(
            self.fc_global2(x_global1) + x_global
        )  # with residual connection before AF

        x_global2local = x_global.view(-1, 1, latent_global).repeat(
            1, n_points, 1
        )  # first add dimension, than expand it
        x_localCATglobal = torch.cat([x_local, x_global2local], 2)

        ### phi p
        x_local1 = F.leaky_relu(
            self.fc_local1(x_localCATglobal)
        )  # with residual connection before AF
        x_local = F.leaky_relu(self.fc_local2(x_local1) + x_local)

        return x_global, x_local


def exclusive_sum_pooling(x):
    # print(f"\n Excl sum pooling x: {x.shape}")
    emb = x.sum(-2, keepdims=True)
    # print(f"\n Excl sum pooling emb: {emb.shape}")
    res = emb - x
    # print(f"\n Excl sum pooling res: {res.shape}")
    return res


def exclusive_mean_pooling(x):

    emb = exclusive_sum_pooling(x)
    N = torch.ones_like(x) * len(x)
    y = emb / torch.max(N - 1, torch.ones_like(N))[0]
    return y


def exclusive_max_pooling(x):
    if x.shape[-2] == 1:  # If only one element in set
        return torch.zeros_like(x)

    first, second = torch.topk(x, 2, dim=-2).values.chunk(2, dim=-2)
    indicator = (x == first).float()
    y = (1 - indicator) * first + indicator * second
    return y


def mean_sum_max_pooling_nn(x):
    mean_pooling = exclusive_mean_pooling(x)
    sum_pooling = exclusive_sum_pooling(x)
    max_pooling = exclusive_max_pooling(x)
    res = torch.cat([mean_pooling, sum_pooling, max_pooling], -1)
    return res


# equivariant layer with global concat & residual connections inside this module  & weight_norm
# ordered: first update global, then local
class DeepSet_layer(nn.Module):
    def __init__(self, in_dim, hid_dim, latent_dim):
        super(DeepSet_layer, self).__init__()

        self.set_emb = MLP(in_dim, [hid_dim, hid_dim], in_dim)
        self.pooling_mlp = MLP(in_dim * 3, [hid_dim, hid_dim], in_dim)
        # in_dim: 128
        # hid_dim: 128
        # latent_dim: 10

    def forward(self, x_global, x_local):
        x = x_local  # (B,P,F)

        x = self.set_emb(x)

        msm_p = mean_sum_max_pooling_nn(x)
        y = self.pooling_mlp(msm_p)
        x_local = y # (B,P,F)

        return x_global, x_local


######################################
###       GENERATOR                ###
######################################


# Decoder / Generator for mutliple particles with Variable Number of Equivariant Layers (with global concat)
# added same global and local usage in EPiC layer
# order: global first, then local
class EPiC_generator(nn.Module):
    def __init__(self, args):
        super(EPiC_generator, self).__init__()
        self.latent = args["latent"]  # used for latent size of equiv concat
        self.latent_local = args["latent_local"]  # noise
        self.hid_d = args["hid_d"]  # default 256
        self.feats = args["feats"]
        self.equiv_layers = args["equiv_layers_generator"]
        self.return_latent_space = args["return_latent_space"]  # false or true

        self.local_0 = weight_norm(nn.Linear(self.latent_local, self.hid_d))
        self.global_0 = weight_norm(nn.Linear(self.latent, self.hid_d))
        self.global_1 = weight_norm(nn.Linear(self.hid_d, self.latent))

        self.nn_list = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.nn_list.append(EPiC_layer(self.hid_d, self.hid_d, self.latent))

        self.local_1 = weight_norm(nn.Linear(self.hid_d, self.feats))

    def forward(self, z_global, z_local):  # shape: [batch, points, feats]
        batch_size, _, _ = z_local.size()
        latent_tensor = z_global.clone().reshape(batch_size, 1, -1)

        z_local = F.leaky_relu(self.local_0(z_local))

        z_global = F.leaky_relu(self.global_0(z_global))
        z_global = F.leaky_relu(self.global_1(z_global))
        latent_tensor = torch.cat(
            [latent_tensor, z_global.clone().reshape(batch_size, 1, -1)], 1
        )

        z_global_in, z_local_in = z_global.clone(), z_local.clone()

        # equivariant connections, each one_hot conditined
        for i in range(self.equiv_layers):
            z_global, z_local = self.nn_list[i](
                z_global, z_local
            )  # contains residual connection
            z_global, z_local = (
                z_global + z_global_in,
                z_local + z_local_in,
            )  # skip connection to sampled input
            latent_tensor = torch.cat(
                [latent_tensor, z_global.clone().reshape(batch_size, 1, -1)], 1
            )

        # final local NN to get down to input feats size
        out = self.local_1(z_local)

        if self.return_latent_space:
            return out, latent_tensor
        else:
            return out  # [batch, points, feats]


class DeepSet_generator(nn.Module):
    def __init__(self, args):
        super(DeepSet_generator, self).__init__()
        self.latent = args["latent"]  # used for latent size of equiv concat
        self.latent_local = args["latent_local"]  # noise
        self.hid_d = args["hid_d"]  # default 256
        self.feats = args["feats"]
        self.equiv_layers = args["equiv_layers_generator"]
        self.return_latent_space = args["return_latent_space"]  # false or true

        self.local_0 = weight_norm(nn.Linear(self.latent_local, self.hid_d))
        self.global_0 = weight_norm(nn.Linear(self.latent, self.hid_d))
        self.global_1 = weight_norm(nn.Linear(self.hid_d, self.latent))

        self.nn_list = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.nn_list.append(DeepSet_layer(self.hid_d, self.hid_d, self.latent))

        self.local_1 = weight_norm(nn.Linear(self.hid_d, self.feats))

    def forward(self, z_global, z_local):  # shape: [batch, points, feats]
        batch_size, _, _ = z_local.size()
        latent_tensor = z_global.clone().reshape(batch_size, 1, -1)

        z_local = F.leaky_relu(self.local_0(z_local))

        # z_global = F.leaky_relu(self.global_0(z_global))
        # z_global = F.leaky_relu(self.global_1(z_global))
        latent_tensor = torch.cat(
            [latent_tensor, z_global.clone().reshape(batch_size, 1, -1)], 1
        )

        z_global_in, z_local_in = z_global.clone(), z_local.clone()

        # equivariant connections, each one_hot conditined
        for i in range(self.equiv_layers):
            z_global, z_local = self.nn_list[i](
                z_global, z_local
            )  # contains residual connection
            z_global, z_local = (
                z_global + z_global_in,
                z_local + z_local_in,
            )  # skip connection to sampled input
            latent_tensor = torch.cat(
                [latent_tensor, z_global.clone().reshape(batch_size, 1, -1)], 1
            )

        # final local NN to get down to input feats size
        out = self.local_1(z_local)

        if self.return_latent_space:
            return out, latent_tensor
        else:
            return out  # [batch, points, feats]


######################################
###       DISCRIMINATOR            ###
######################################


# Discriminator: Deep Sets like 3 + 3 layer with residual connections  & weight_norm   & mix(mean/sum/max) pooling  & NO multipl. cond.
class EPiC_discriminator(nn.Module):
    def __init__(self, args):
        super(EPiC_discriminator, self).__init__()
        self.hid_d = args["hid_d"]
        self.feats = args["feats"]
        self.equiv_layers = args["equiv_layers_discriminator"]
        self.latent = args["latent"]  # used for latent size of equiv concat

        self.fc_l1 = weight_norm(nn.Linear(self.feats, self.hid_d))
        self.fc_l2 = weight_norm(nn.Linear(self.hid_d, self.hid_d))

        self.fc_g1 = weight_norm(nn.Linear(int(2 * self.hid_d), self.hid_d))
        self.fc_g2 = weight_norm(nn.Linear(self.hid_d, self.latent))

        self.nn_list = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.nn_list.append(EPiC_layer(self.hid_d, self.hid_d, self.latent))

        self.fc_g3 = weight_norm(
            nn.Linear(int(2 * self.hid_d + self.latent), self.hid_d)
        )
        self.fc_g4 = weight_norm(nn.Linear(self.hid_d, self.hid_d))
        self.fc_g5 = weight_norm(nn.Linear(self.hid_d, 1))

    def forward(self, x):
        # local encoding
        x_local = F.leaky_relu(self.fc_l1(x))
        x_local = F.leaky_relu(self.fc_l2(x_local) + x_local)

        # global features
        x_mean = x_local.mean(1, keepdim=False)  # mean over points dim.
        x_sum = x_local.sum(1, keepdim=False)  # mean over points dim.
        x_global = torch.cat([x_mean, x_sum], 1)
        x_global = F.leaky_relu(self.fc_g1(x_global))
        x_global = F.leaky_relu(self.fc_g2(x_global))  # projecting down to latent size

        # equivariant connections
        for i in range(self.equiv_layers):
            x_global, x_local = self.nn_list[i](
                x_global, x_local
            )  # contains residual connection

        x_mean = x_local.mean(1, keepdim=False)  # mean over points dim.
        x_sum = x_local.sum(1, keepdim=False)  # sum over points dim.
        x = torch.cat([x_mean, x_sum, x_global], 1)

        x = F.leaky_relu(self.fc_g3(x))
        x = F.leaky_relu(self.fc_g4(x) + x)
        x = self.fc_g5(x)
        return x


# Discriminator: Deep Sets like 3 + 3 layer with residual connections  & weight_norm   & mix(mean/sum/max) pooling  & NO multipl. cond.
class DeepSet_discriminator(nn.Module):
    def __init__(self, args):
        super(DeepSet_discriminator, self).__init__()
        self.hid_d = args["hid_d"]
        self.feats = args["feats"]
        self.equiv_layers = args["equiv_layers_discriminator"]
        self.latent = args["latent"]  # used for latent size of equiv concat

        self.fc_l1 = weight_norm(nn.Linear(self.feats, self.hid_d))
        self.fc_l2 = weight_norm(nn.Linear(self.hid_d, self.hid_d))

        # self.fc_g1 = weight_norm(nn.Linear(int(2 * self.hid_d), self.hid_d))
        # self.fc_g2 = weight_norm(nn.Linear(self.hid_d, self.latent))

        self.nn_list = nn.ModuleList()
        for _ in range(self.equiv_layers):
            self.nn_list.append(DeepSet_layer(self.hid_d, self.hid_d, self.latent))

        self.fc_g3 = weight_norm(nn.Linear(int(2 * self.hid_d), self.hid_d))
        self.fc_g4 = weight_norm(nn.Linear(self.hid_d, self.hid_d))
        self.fc_g5 = weight_norm(nn.Linear(self.hid_d, 1))

    def forward(self, x):  # (B,P,F)
        # local encoding
        x_local = F.leaky_relu(self.fc_l1(x))
        x_local = F.leaky_relu(self.fc_l2(x_local) + x_local)  # (B,P,hid_d)

        # global features
        x_mean = x_local.mean(1, keepdim=False)  # mean over points dim. # (B,P)
        x_sum = x_local.sum(1, keepdim=False)  # mean over points dim. # (B,P)
        x_global = x  # (B,P,F)
        # torch.cat([x_mean, x_sum], 1)
        # x_global = F.leaky_relu(self.fc_g1(x_global))
        # x_global = F.leaky_relu(self.fc_g2(x_global))  # projecting down to latent size

        # equivariant connections
        for i in range(self.equiv_layers):
            x_global, x_local = self.nn_list[i](
                x_global, x_local
            )  # contains residual connection # (B,P,F), (B,P,hid_d)

        x_mean = x_local.mean(1, keepdim=False)  # mean over points dim. # (B,hid_d)
        x_sum = x_local.sum(1, keepdim=False)  # sum over points dim. # (B,hid_d)

        x = torch.cat([x_mean, x_sum], 1)  # (B,2*hid_d)
        x = F.leaky_relu(self.fc_g3(x))
        x = F.leaky_relu(self.fc_g4(x) + x)
        x = self.fc_g5(x)  # (B,1)
        return x
