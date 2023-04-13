import torch
import torch.nn as nn


class DaviesBouldinLossFunction(nn.Module):
    def __init__(self, json=None):
        super(DaviesBouldinLossFunction, self).__init__()
        self.distances = None
        self.sum = None
        self.count = None
        self.centroids = None
        self.class_number = self.get_class_number(json)
        self.epoch = 0

    def init_tensors(self, predicted, target):
        out_dimension = predicted.shape[1]
        self.sum = torch.zeros(self.class_number, out_dimension)
        self.count = torch.zeros(self.class_number, 1)
        self.centroids = torch.zeros(self.class_number, 1)

        self.distances = torch.zeros(self.class_number, 1)




    def update_centroids(self, predicted, target):
        self.sum.index_add_(0, target, predicted.float())
        count = torch.bincount(target)
        count = torch.nn.functional.pad(count, pad=(0, self.class_number - count.shape[0]))
        self.count[:, 0] += count

    def update_distances(self, predicted, target):
        t = torch.index_select(self.centroids, 0, target)
        t = torch.norm(t - predicted, dim=1)
        self.distances.index_add_(0, target, t)

    def calculate_centroids(self):
        self.centroids = self.sum/self.count
        #self.sum.zero_()

    @staticmethod
    def get_class_number(json):
        class_number = 0
        for i in json.keys():
            class_number += len(json[i])
        return class_number

    def calculate_coherence(self, predicted, target):
        t = torch.index_select(self.centroids, 0, target)
        t = torch.norm(t - predicted, dim = 1)
        return t

    def calculate_separation(self, predicted, target):
        centroids = self.centroids.detach().clone()
        s = self.distances.detach().clone()

        counts = torch.index_select(self.count, 0, target)
        centroids.index_add_(0, target, predicted/counts)
        cnts = torch.index_select(centroids, 0, target)
        pr = predicted/counts
        vec = torch.norm(cnts - pr, dim=1)
        '''if vec.shape[0] < -1:#self.class_number:
            vec = torch.nn.functional.pad(vec, pad=(0, self.class_number - vec.shape[0]))
            target_ = torch.nn.functional.pad(target, pad=(0, self.class_number - target.shape[0]))
            vec = vec.reshape(self.class_number, 1)
            s.index_add_(0, target_, vec.expand_as(s).contiguous())
        else:'''
        s.index_add_(0, target, vec.reshape(target.shape[0], 1))



        #for p, t, index in zip(predicted, target, range(0, target.shape[0])):
        #    s[t.item()] += torch.norm(centroids[t.item()] - p/self.count[t.item()], dim=0)
        s = torch.sqrt(s)
        s = s/self.count
        m = torch.cdist(centroids, centroids, p=2)
        sum = torch.zeros(1)
        for i in range(0, self.class_number):
            for j in range(0, self.class_number):
                if i != j:
                    sum += (s[i] + s[j])/m[i][j]
        return sum/self.class_number






    def forward(self, predicted, target, epoch):
        if self.sum is None:
            self.init_tensors(predicted, target)
        if self.epoch != epoch:
            if epoch in [0, 3, 6, 9, 12, 15]:
                self.init_tensors(predicted, target)
            self.epoch = epoch
        if epoch % 3 == 0:
            self.update_centroids(predicted, target)
        elif epoch % 3 == 1:
            self.calculate_centroids()
            self.update_distances(predicted, target)
        elif epoch % 3 == 2:
            loss = self.calculate_separation(predicted, target)
            if epoch == 5:
                pass
                #print(predicted, target)
                #print(self.centroids)
            return loss

        loss = torch.norm(predicted[:, 0] - predicted[:, 0])
        return loss
