import torch
import tensorflow as tf
class DIST_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = torch.zeros(1)
        for i in range(pred.shape[0]):
            p = pred[i]
            t = target[i]
            dist = (torch.pow((p[0]-t[0]),2) + torch.pow((p[1]-t[1]),2)).sqrt().unsqueeze(0)

            loss = torch.cat((loss,dist),0)
        loss = loss.mean()
        return loss


def a():
    d_res4b = []
    d_res4b.extend([1, 2, 5, 9] * 5 + [1, 2, 5])
    print(d_res4b)

def b():
    label = torch.as_tensor([[0.4588, 0.4437],
                             [0.8975, 0.6037],
                             [0.5263, 0.5525],
                             [0.6338, 0.6475],
                             [0.2975, 0.4812],
                             [0.4775, 0.5450],
                             [0.6900, 0.3787],
                             [0.5550, 0.4200]])

    pred = torch.as_tensor(([[-4.1203, -1.2986],
                             [-4.2784, -1.3029],
                             [-5.4924, -1.5659],
                             [-5.3747, -0.8996],
                             [-3.5245, -1.4163],
                             [-5.8750, -1.7695],
                             [-5.0309, -1.0557],
                             [-4.8352, -1.1514]]))

    loss = DIST_loss()
    loss1 = torch.nn.L1Loss()
    loss2 = tf.nn.l2_loss()
    c = loss(pred, label)
    d = loss1(pred, label)
    e = loss2(pred, label)
    print(c, d, e)
if __name__ == '__main__':
   a()