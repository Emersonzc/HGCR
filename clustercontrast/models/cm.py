import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class CM_Hybrid(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm_hybrid(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))



def cm_hybrid_v2(inputs, indexes, features,momentum=0.5):
    return CM_Hybrid_v2.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class CM_Hybrid_v2(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features,  momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index+nums] = ctx.features[index+nums] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index+nums] /= ctx.features[index+nums].norm()

        return grad_inputs, None, None, None


def cm_hybrid_v3(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid_v3.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class CM_Hybrid_v3(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features,  momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index+nums] = ctx.features[index+nums] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index+nums] /= ctx.features[index+nums].norm()

        return grad_inputs, None, None, None

def cm_hybrid_v4(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid_v4.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class CM_Hybrid_v4(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        # ctx.num_cluster = num_cluster
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//3
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

            max = np.argmax(np.array(distances))
            ctx.features[index + nums] = ctx.features[index + nums] * ctx.momentum/2 + (1 - ctx.momentum/2) * features[max]
            ctx.features[index + nums] /= ctx.features[index + nums].norm()

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index+nums+nums] = ctx.features[index+nums+nums] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index+nums+nums] /= ctx.features[index+nums+nums].norm()

        return grad_inputs, None, None, None

def cm_hybrid_v5(inputs, indexes, features, num_cluster,momentum=0.5):
    return CM_Hybrid_v5.apply(inputs, indexes, features,num_cluster, torch.Tensor([momentum]).to(inputs.device))

class CM_Hybrid_v5(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, num_cluster, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.num_cluster = num_cluster
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            if index > ctx.num_cluster:
                median = np.argmax(np.array(distances))
                ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
                ctx.features[index] /= ctx.features[index].norm()
            else:
                mean = torch.stack(features, dim=0).mean(0)
                ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean
                ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None

class CM_Hybrid_v6(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            if index > ctx.num_cluster:
                max = np.argmax(np.array(distances))
                ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[max]
                ctx.features[index] /= ctx.features[index].norm()
            else:
                median = np.argmin(np.array(distances))
                ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
                ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hybrid_v6(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid_v6.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class ClusterMemory(nn.Module, ABC):
    __CMfactory = {
        'CM': cm,
        'CMhard': cm_hard,
    }

    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, mode='CM', hard_weight=0.5):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        # self.use_hard = use_hard
        self.momentum = momentum
        self.temp = temp
        self.cm_type = mode
        self.cross_entropy = nn.CrossEntropyLoss().cuda()

        if self.cm_type in ['CM', 'CMhard']:
            self.register_buffer('features', torch.zeros(num_samples, num_features))
        elif self.cm_type == 'CMhybrid':
            self.hard_weight = hard_weight
            print('hard_weight: {}'.format(self.hard_weight))
            self.register_buffer('features', torch.zeros(num_samples, num_features))
        elif self.cm_type == 'CMhybrid_v2':
            self.hard_weight = hard_weight
            print('hard_weight: {}'.format(self.hard_weight))
            self.register_buffer('features', torch.zeros(2*num_samples, num_features))
        elif self.cm_type == 'CMhybrid_v3':
            self.hard_weight = hard_weight
            self.register_buffer('features', torch.zeros(2*num_samples, num_features))
        elif self.cm_type == 'CMhybrid_v4':
            self.hard_weight = hard_weight
            self.register_buffer('features', torch.zeros(3*num_samples, num_features))
        elif self.cm_type == 'CMhybrid_v5':
            self.hard_weight = hard_weight
            self.register_buffer('features', torch.zeros(num_samples, num_features))
        else:
            raise TypeError('Cluster Memory {} is invalid!'.format(self.cm_type))

    def forward(self, inputs, targets):

        if self.cm_type in ['CM', 'CMhard']:
            outputs = ClusterMemory.__CMfactory[self.cm_type](inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            loss = self.cross_entropy(outputs, targets)
            return loss

        elif self.cm_type=='CMhybrid':
            outputs = cm_hybrid(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            loss = self.cross_entropy(outputs, targets)
            return loss

        elif self.cm_type == 'CMhybrid_v2':
            outputs = cm_hybrid_v2(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            output_hard, output_mean = torch.chunk(outputs, 2, dim=1)
            loss = self.hard_weight * self.cross_entropy(output_hard, targets) + (1 - self.hard_weight) * self.cross_entropy(output_mean, targets)
            return loss
        elif self.cm_type == 'CMhybrid_v3':
            outputs = cm_hybrid_v3(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            output_hard, output_mean = torch.chunk(outputs, 2, dim=1)
            loss = self.hard_weight * self.cross_entropy(output_hard, targets) + (1 - self.hard_weight) * self.cross_entropy(output_mean, targets)
            return loss
        elif self.cm_type == 'CMhybrid_v4':
            outputs = cm_hybrid_v4(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            output_hard_neg, output_hard_pos, output_mean = torch.chunk(outputs, 3, dim=1)
            loss = self.hard_weight * (self.cross_entropy(output_hard_pos, targets)/2 +
                    self.cross_entropy(output_hard_neg, targets)/2 + (1 - self.hard_weight) * self.cross_entropy(
                output_mean, targets))
            return loss
        elif self.cm_type == 'CMhybrid_v5':
            outputs = cm_hybrid_v5(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            loss = self.cross_entropy(outputs, targets)
            return loss
