import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        if os.path.exists('./model/model.pth'):
            self.load_state_dict(torch.load('./model/model.pth'))
            print('Model loaded')

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criteria = nn.MSELoss()

    def train_step(self, old_state, move, reward, new_state, done):
        old_state = numpy.array(old_state)
        old_state = torch.tensor(old_state, dtype=torch.float)
        new_state = numpy.array(new_state)
        new_state = torch.tensor(new_state, dtype=torch.float)
        move = numpy.array(move)
        move = torch.tensor(move, dtype=torch.long)
        reward = numpy.array(reward)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(old_state.shape) == 1:
            old_state = torch.unsqueeze(old_state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            move = torch.unsqueeze(move, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(old_state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(new_state[idx]))

            target[idx][torch.argmax(move[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criteria(target, pred)
        loss.backward()
        self.optimizer.step()
