import torch
import torch.nn as nn

def add_eps(x, eps=1e-7):
    EPS = torch.full_like(x, eps)
    EPS[x < 0] = -eps
    return torch.add(x, EPS)

class Std(nn.Module):
    def __init__(self, stride=10) -> None:
        super().__init__()
        self.stride = stride
    
    def forward(self, feature):
        B, L, D = feature.shape

        feature = feature.reshape(-1, self.stride, D)
        return torch.std(feature, dim=-2, unbiased=False).reshape(-1, int(L/self.stride), D)


class ZScore(nn.Module):
    def __init__(self, stride=10) -> None:
        super().__init__()
        self.stride = stride
    
    def forward(self, feature):
        B, L, D = feature.shape

        feature = feature.reshape(B, -1, self.stride, D)
        std = torch.std(feature, dim=-2, unbiased=False)
        std = add_eps(std)
        mean = torch.mean(feature, dim=-2)
        
        z_score = mean / std

        return z_score


class Return(nn.Module):
    def __init__(self, stride=10) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, feature):

        numerators = feature[:, (self.stride - 1)::self.stride, :]
        denominators = feature[:, 0::self.stride, :]
        denominators = add_eps(denominators)
        
        return numerators / (denominators) - 1.0


class LinearDecay(nn.Module):
    def __init__(self, stride) -> None:
        super().__init__()
        self.stride = stride
        lineardecay = torch.linspace(1.0, self.stride, steps=self.stride)
        self.weights =  lineardecay / lineardecay.sum()
    
    def forward(self, feature):
        B, L, D = feature.shape
        feature = feature.reshape(B, -1, self.stride, D)
        weights = self.weights.unsqueeze(1).expand(self.stride, D).to(feature)
        weighted_feature = feature * weights

        return torch.sum(weighted_feature, dim=-2)


class Covariance(nn.Module):
    def __init__(self, stride) -> None:
        super().__init__()
        self.stride = stride
        self.intermediate_shape = None
        self.out_shape = None
        self.lower_mask = None
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(self.stride, 1), stride=(self.stride, 1))
    
    def forward(self, feature):
        B, L, D = feature.shape
        self.intermediate_shape = (-1, self.stride, D)
        output_features = int(D * (D - 1) / 2)
        self.out_shape = (-1, int(L/self.stride), output_features)
        self.lower_mask = torch.zeros(D, D, dtype=torch.int32).to(feature)
        for i in range(D):
            self.lower_mask[i+1:, i] = 1
        
        means = self.avg_pool2d(feature[:, None, :, :]).squeeze(dim=1)
        means_brodcast = torch.repeat_interleave(means, repeats=self.stride, dim=1)
        means_substracted = torch.subtract(feature, means_brodcast)
        means_substracted = means_substracted.reshape(self.intermediate_shape)
        
        covariance_matrix = torch.einsum("ijk, ijm->ikm",
                                         means_substracted,
                                         means_substracted)

        covariance_matrix = covariance_matrix / (self.stride - 1)

        covariances = torch.masked_select(covariance_matrix, self.lower_mask.bool())
        covariances = covariances.reshape(self.out_shape)
        return covariances


class Correlation(nn.Module):
    def __init__(self, stride) -> None:
        super().__init__()
        self.stride = stride
        self.intermediate_shape = None
        self.out_shape = None
        self.lower_mask = None
        self.avg_pool2d = nn.AvgPool2d(kernel_size=(self.stride, 1), stride=(self.stride, 1))
    
    def forward(self, feature):
        B, L, D = feature.shape
        self.intermediate_shape = (-1, self.stride, D)
        output_features = int(D * (D - 1) / 2)
        self.out_shape = (-1, int(L/self.stride), output_features)
        self.lower_mask = torch.zeros(D, D, dtype=torch.int32).to(feature)
        for i in range(D):
            self.lower_mask[i+1:, i] = 1
        
        means = self.avg_pool2d(feature[:, None, :, :]).squeeze(dim=1)
        means_brodcast = torch.repeat_interleave(means, repeats=self.stride, dim=1)
        means_substracted = torch.subtract(feature, means_brodcast)
        means_substracted = means_substracted.reshape(self.intermediate_shape)

        squared_diff = torch.square(means_substracted)
        mean_squared_error = torch.mean(squared_diff, dim=1)
        std = torch.sqrt(mean_squared_error)

        # get denominator of correlation matrix
        denominator_matrix = torch.einsum("ik,im->ikm", std, std)

        # compute covariance matrix
        covariance_matrix = torch.einsum("ijk,ijm->ikm",
                                       means_substracted,
                                       means_substracted)
        covariance_matrix = covariance_matrix / self.stride

        covariances = torch.masked_select(covariance_matrix, self.lower_mask.bool())
        denominators = torch.masked_select(denominator_matrix, self.lower_mask.bool())
        denominators = add_eps(denominators)
        correlations = torch.div(covariances, denominators)
        correlations = torch.reshape(correlations, self.out_shape)

        return correlations


class FeatureExpansion(nn.Module):
    def __init__(self, stride=10) -> None:
        super().__init__()
        self.stride = stride
        self.std = Std(stride=self.stride)
        self.z_score = ZScore(stride=self.stride)
        self.linear_decay = LinearDecay(stride=self.stride)
        self.return_rate = Return(stride=self.stride)
        self.covariance = Covariance(stride=self.stride)
        self.correlation = Correlation(stride=self.stride)
    
    def forward(self, inputs):
        std_output = self.std(inputs)
        z_score_output = self.z_score(inputs)
        decay_linear_output = self.linear_decay(inputs)
        return_output = self.return_rate(inputs)
        covariance_output = self.covariance(inputs)
        correlation_output = self.correlation(inputs)

        return torch.cat([std_output,
                           z_score_output,
                           decay_linear_output,
                           return_output,
                           covariance_output,
                           correlation_output], dim=2)


class AlphaNetV3(nn.Module):
    def __init__(self, feat_dim=6, hidden_dim=30, num_layers=1, dropout=0.0, num_classes=3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.expand_dim = feat_dim * (feat_dim + 3)
        self.expanded10 = FeatureExpansion(stride=10)
        self.expanded5 = FeatureExpansion(stride=5)
        # expanded return shape (B, L/strie, feature*(feature+3))
        self.bn10 = nn.BatchNorm1d(self.expand_dim)
        self.bn5 = nn.BatchNorm1d(self.expand_dim)

        self.recurrent10 = nn.LSTM(input_size=self.expand_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.recurrent5 = nn.LSTM(input_size=self.expand_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        self.bn10_2 = nn.BatchNorm1d(hidden_dim)
        self.bn5_2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if num_classes == 1:
            self.outputs = nn.Linear(hidden_dim*2, 1)
        else:
            self.outputs = nn.Linear(hidden_dim*2, num_classes)
            # self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        expanded10 = self.expanded10(inputs)
        expanded5 = self.expanded5(inputs)

        bn10 = self.bn10(expanded10.transpose(1, 2)).transpose(1, 2)
        bn5 = self.bn5(expanded5.transpose(1, 2)).transpose(1, 2)
        
        recurrent10 = self.recurrent10(bn10)[0][:, -1, :]
        recurrent5 = self.recurrent5(bn5)[0][:, -1, :]

        bn10_2 = self.bn10_2(recurrent10)
        bn5_2 = self.bn5_2(recurrent5)

        x = torch.cat([bn10_2, bn5_2], dim=1)
        x = self.dropout(x)
        out = self.outputs(x)
        # if self.num_classes > 1:
        #     out = self.softmax(out)

        return out