import time
import pickle
import numpy as np
import autograd_hacks

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x = pickle.load(open('x_MNIST', 'rb'))
x = np.expand_dims(x, axis=1)
x = torch.Tensor(x)
N = len(x)

y = pickle.load(open('y_MNIST', 'rb'))
y = torch.LongTensor(y)

x_test = pickle.load(open('x_test_MNIST', 'rb'))
x_test = np.expand_dims(x_test, axis=1)
x_test = torch.Tensor(x_test)
N_test = len(x_test)

y_test = pickle.load(open('y_test_MNIST', 'rb'))
y_test = torch.LongTensor(y_test)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.fc1 = nn.Linear(4000, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 4000)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
    
    

class AbstractRegressor():
    
    def __init__(self, lr, batch_size, optimizer_name, x, y, x_test, y_test):
        """Initialize a Regressor object.
        """
        
        self.lr = lr
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test

        self.model = Net()
        self.loss_function = nn.NLLLoss(reduction='none')
        
        if optimizer_name == 'RMSP':
            self.model_optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'ADAM':
            self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'SGD_M':
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        else:
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
    
    def learn(self):
        raise NotImplementedError()


    def evaluate_regressor(self):
        with torch.no_grad():
            y_hat_test = self.model(x_test)
            loss_test = self.loss_function(y_hat_test, y_test).mean()
            y_hat_pred = torch.argmax(y_hat_test, dim=1)
        correct = ((y_hat_pred.numpy() - y_test.numpy()) ** 2).mean()
        return correct, loss_test



class UniformRegressor(AbstractRegressor):
    
    def __init__(self, lr, batch_size, optimizer_name, x, y, x_test, y_test):
        super(AbstractRegressor, self).__init__()
        
        self.lr = lr
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.N = len(self.y)
        self.x_test = x_test
        self.y_test = y_test
        self.N_test = len(self.y_test)

        self.model = Net()
        self.loss_function = nn.NLLLoss(reduction='none')
        
        if optimizer_name == 'RMSP':
            self.model_optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'ADAM':
            self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'SGD_M':
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        else:
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        
    
    def learn(self):
        K = np.random.choice(N, size=self.batch_size)
        K = torch.LongTensor(K)
        x_for_batch = self.x[K]
        y_for_batch = self.y[K]
    
        y_hat = self.model(x_for_batch)
        
        loss = self.loss_function(y_hat, y_for_batch).mean()
    
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()






class OptimalRegressor(AbstractRegressor):
    
    def __init__(self, lr, batch_size, optimizer_name, x, y, x_test, y_test):
        super(AbstractRegressor, self).__init__()
        
        self.lr = lr
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.N = len(self.y)
        self.x_test = x_test
        self.y_test = y_test
        self.N_test = len(self.y_test)

        self.model = Net()
        self.loss_function = nn.NLLLoss(reduction='none')
        
        if optimizer_name == 'RMSP':
            self.model_optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'ADAM':
            self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'SGD_M':
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        else:
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        
    
    def learn(self):
        autograd_hacks.add_hooks(self.model)
        y_hat = self.model(self.x)
    
        loss = self.loss_function(y_hat, self.y).mean()
    
    
    
        self.model_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        autograd_hacks.compute_grad1(self.model)
        autograd_hacks.disable_hooks()
        autograd_hacks.clear_backprops(self.model)
        autograd_hacks.remove_hooks(self.model)
    
        grad_norms1 = np.zeros(N)
        for layer in self.model.modules():
            if not autograd_hacks.is_supported(layer):
                continue
            for param in layer.parameters():
                for i in range(N):
                    grad_norms1[i] += np.linalg.norm(param.grad1[i].numpy()) ** 2
        grad_norms = np.sqrt(grad_norms1) + 1e-15
    
        priorities_prob = grad_norms / grad_norms.sum()
        indices = np.random.choice(np.arange(self.N), self.batch_size, p=priorities_prob)
        x_for_batch = x[indices]
        y_for_batch = y[indices]
        grad_norms_selected = grad_norms[indices]
    
        y_hat_for_batch = self.model(x_for_batch)
    
        loss_weights = grad_norms.mean() / grad_norms_selected
        loss_weights = torch.from_numpy(loss_weights).unsqueeze(1)
        loss = loss_weights * self.loss_function(y_hat_for_batch, y_for_batch)
        loss = loss.mean()
    
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()






class SemiOptimalRegressor(AbstractRegressor):
    
    def __init__(self, lr, batch_size, optimizer_name, x, y, x_test, y_test):
        super(AbstractRegressor, self).__init__()
        
        self.lr = lr
        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.N = len(self.y)
        self.x_test = x_test
        self.y_test = y_test
        self.N_test = len(self.y_test)

        self.model = Net()
        self.loss_function = nn.NLLLoss(reduction='none')
        
        if optimizer_name == 'RMSP':
            self.model_optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'ADAM':
            self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer_name == 'SGD_M':
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5)
        else:
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        
    
    def learn(self):
        autograd_hacks.add_hooks(self.model)
        y_hat = self.model(self.x)
    
        loss = self.loss_function(y_hat, self.y).mean()
    
    
    
        self.model_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        autograd_hacks.compute_grad1(self.model)
        autograd_hacks.disable_hooks()
        autograd_hacks.clear_backprops(self.model)
        autograd_hacks.remove_hooks(self.model)
    
        grad_norms1 = np.zeros(N)
        for layer in self.model.modules():
            if not autograd_hacks.is_supported(layer):
                continue
            for param in layer.parameters():
                for i in range(N):
                    grad_norms1[i] += np.linalg.norm(param.grad1[i].numpy()) ** 2
        grad_norms = np.sqrt(grad_norms1) + 1e-15
    
        priorities_prob = grad_norms / grad_norms.sum()
        unif = np.ones(N) / N
        probs = (priorities_prob + unif) / 2
        
        indices = np.random.choice(np.arange(self.N), self.batch_size, p=probs)
        x_for_batch = x[indices]
        y_for_batch = y[indices]
        probs_selected = probs[indices]
    
        y_hat_for_batch = self.model(x_for_batch)
    
        loss_weights = 1 / (N * probs_selected)
        loss_weights = torch.from_numpy(loss_weights).unsqueeze(1)
        loss = loss_weights * self.loss_function(y_hat_for_batch, y_for_batch)
        loss = loss.mean()
    
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()












batch_size = 5
nb_steps = 350 # number of gradient steps
nb_runs = 45 # number of runs to average results
lr = 1e-2



#############################################
# Uniform sampling
#############################################

regressor_name = 'UniformRegressor'

optimizer_name = 'SGD'
list_time = []
t_start = time.time()
for i in range(nb_runs):
    t_start_run = time.time()
    regressor = UniformRegressor(lr, batch_size, optimizer_name, x, y, x_test, y_test)
    losses = []
    accuracy = []
    for j in range(nb_steps):
        regressor.learn()
        correct_on_test_set, loss_on_test_set = regressor.evaluate_regressor()
        losses.append(loss_on_test_set)
        accuracy.append(correct_on_test_set)
        if j % 50 == 0 :
            print(j)
    t_end_run = time.time()
    list_time.append(t_end_run - t_start_run)
    title1 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_loss_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title1, 'wb'))
    title2 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_acc_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title2, 'wb'))
t_end = time.time()
pickle.dump(list_time, open('results/' + regressor_name + '_' + optimizer_name + '_list_time', 'wb'))
to_print = 'For ' + str(nb_runs) + ' runs, optimization with ' + regressor_name + ' and ' + optimizer_name + ' took ' + str(t_end-t_start) + ' seconds, which yields ' + str((t_end-t_start)/nb_runs) + ' seconds per run.'
print(to_print)


optimizer_name = 'SGD_M'
list_time = []
t_start = time.time()
for i in range(nb_runs):
    t_start_run = time.time()
    regressor = UniformRegressor(lr, batch_size, optimizer_name, x, y, x_test, y_test)
    losses = []
    accuracy = []
    for j in range(nb_steps):
        regressor.learn()
        correct_on_test_set, loss_on_test_set = regressor.evaluate_regressor()
        losses.append(loss_on_test_set)
        accuracy.append(correct_on_test_set)
        if j % 50 == 0 :
            print(j)
    t_end_run = time.time()
    list_time.append(t_end_run - t_start_run)
    title1 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_loss_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title1, 'wb'))
    title2 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_acc_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title2, 'wb'))
t_end = time.time()
pickle.dump(list_time, open('results/' + regressor_name + '_' + optimizer_name + '_list_time', 'wb'))
to_print = 'For ' + str(nb_runs) + ' runs, optimization with ' + regressor_name + ' and ' + optimizer_name + ' took ' + str(t_end-t_start) + ' seconds, which yields ' + str((t_end-t_start)/nb_runs) + ' seconds per run.'
print(to_print)



optimizer_name = 'RMSP'
list_time = []
t_start = time.time()
for i in range(nb_runs):
    t_start_run = time.time()
    regressor = UniformRegressor(lr, batch_size, optimizer_name, x, y, x_test, y_test)
    losses = []
    accuracy = []
    for j in range(nb_steps):
        regressor.learn()
        correct_on_test_set, loss_on_test_set = regressor.evaluate_regressor()
        losses.append(loss_on_test_set)
        accuracy.append(correct_on_test_set)
        if j % 50 == 0 :
            print(j)
    t_end_run = time.time()
    list_time.append(t_end_run - t_start_run)
    title1 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_loss_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title1, 'wb'))
    title2 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_acc_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title2, 'wb'))
t_end = time.time()
pickle.dump(list_time, open('results/' + regressor_name + '_' + optimizer_name + '_list_time', 'wb'))
to_print = 'For ' + str(nb_runs) + ' runs, optimization with ' + regressor_name + ' and ' + optimizer_name + ' took ' + str(t_end-t_start) + ' seconds, which yields ' + str((t_end-t_start)/nb_runs) + ' seconds per run.'
print(to_print)



optimizer_name = 'ADAM'
list_time = []
t_start = time.time()
for i in range(nb_runs):
    t_start_run = time.time()
    regressor = UniformRegressor(lr, batch_size, optimizer_name, x, y, x_test, y_test)
    losses = []
    accuracy = []
    for j in range(nb_steps):
        regressor.learn()
        correct_on_test_set, loss_on_test_set = regressor.evaluate_regressor()
        losses.append(loss_on_test_set)
        accuracy.append(correct_on_test_set)
        if j % 50 == 0 :
            print(j)
    t_end_run = time.time()
    list_time.append(t_end_run - t_start_run)
    title1 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_loss_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title1, 'wb'))
    title2 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_acc_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title2, 'wb'))
t_end = time.time()
pickle.dump(list_time, open('results/' + regressor_name + '_' + optimizer_name + '_list_time', 'wb'))
to_print = 'For ' + str(nb_runs) + ' runs, optimization with ' + regressor_name + ' and ' + optimizer_name + ' took ' + str(t_end-t_start) + ' seconds, which yields ' + str((t_end-t_start)/nb_runs) + ' seconds per run.'
print(to_print)











#############################################
# Optimal sampling
#############################################

regressor_name = 'OptimalRegressor'

optimizer_name = 'SGD'
list_time = []
t_start = time.time()
for i in range(nb_runs):
    t_start_run = time.time()
    regressor = OptimalRegressor(lr, batch_size, optimizer_name, x, y, x_test, y_test)
    losses = []
    accuracy = []
    for j in range(nb_steps):
        regressor.learn()
        correct_on_test_set, loss_on_test_set = regressor.evaluate_regressor()
        losses.append(loss_on_test_set)
        accuracy.append(correct_on_test_set)
        if j % 50 == 0 :
            print(j)
    t_end_run = time.time()
    list_time.append(t_end_run - t_start_run)
    title1 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_loss_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title1, 'wb'))
    title2 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_acc_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title2, 'wb'))
t_end = time.time()
pickle.dump(list_time, open('results/' + regressor_name + '_' + optimizer_name + '_list_time', 'wb'))
to_print = 'For ' + str(nb_runs) + ' runs, optimization with ' + regressor_name + ' and ' + optimizer_name + ' took ' + str(t_end-t_start) + ' seconds, which yields ' + str((t_end-t_start)/nb_runs) + ' seconds per run.'
print(to_print)



optimizer_name = 'SGD_M'
list_time = []
t_start = time.time()
for i in range(nb_runs):
    t_start_run = time.time()
    regressor = OptimalRegressor(lr, batch_size, optimizer_name, x, y, x_test, y_test)
    losses = []
    accuracy = []
    for j in range(nb_steps):
        regressor.learn()
        correct_on_test_set, loss_on_test_set = regressor.evaluate_regressor()
        losses.append(loss_on_test_set)
        accuracy.append(correct_on_test_set)
        if j % 50 == 0 :
            print(j)
    t_end_run = time.time()
    list_time.append(t_end_run - t_start_run)
    title1 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_loss_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title1, 'wb'))
    title2 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_acc_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title2, 'wb'))
t_end = time.time()
pickle.dump(list_time, open('results/' + regressor_name + '_' + optimizer_name + '_list_time', 'wb'))
to_print = 'For ' + str(nb_runs) + ' runs, optimization with ' + regressor_name + ' and ' + optimizer_name + ' took ' + str(t_end-t_start) + ' seconds, which yields ' + str((t_end-t_start)/nb_runs) + ' seconds per run.'
print(to_print)





optimizer_name = 'RMSP'
list_time = []
t_start = time.time()
for i in range(nb_runs):
    t_start_run = time.time()
    regressor = OptimalRegressor(lr, batch_size, optimizer_name, x, y, x_test, y_test)
    losses = []
    accuracy = []
    for j in range(nb_steps):
        regressor.learn()
        correct_on_test_set, loss_on_test_set = regressor.evaluate_regressor()
        losses.append(loss_on_test_set)
        accuracy.append(correct_on_test_set)
        if j % 50 == 0 :
            print(j)
    t_end_run = time.time()
    list_time.append(t_end_run - t_start_run)
    title1 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_loss_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title1, 'wb'))
    title2 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_acc_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title2, 'wb'))
t_end = time.time()
pickle.dump(list_time, open('results/' + regressor_name + '_' + optimizer_name + '_list_time', 'wb'))
to_print = 'For ' + str(nb_runs) + ' runs, optimization with ' + regressor_name + ' and ' + optimizer_name + ' took ' + str(t_end-t_start) + ' seconds, which yields ' + str((t_end-t_start)/nb_runs) + ' seconds per run.'
print(to_print)



optimizer_name = 'ADAM'
list_time = []
t_start = time.time()
for i in range(nb_runs):
    t_start_run = time.time()
    regressor = OptimalRegressor(lr, batch_size, optimizer_name, x, y, x_test, y_test)
    losses = []
    accuracy = []
    for j in range(nb_steps):
        regressor.learn()
        correct_on_test_set, loss_on_test_set = regressor.evaluate_regressor()
        losses.append(loss_on_test_set)
        accuracy.append(correct_on_test_set)
        if j % 50 == 0 :
            print(j)
    t_end_run = time.time()
    list_time.append(t_end_run - t_start_run)
    title1 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_loss_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title1, 'wb'))
    title2 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_acc_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title2, 'wb'))
t_end = time.time()
pickle.dump(list_time, open('results/' + regressor_name + '_' + optimizer_name + '_list_time', 'wb'))
to_print = 'For ' + str(nb_runs) + ' runs, optimization with ' + regressor_name + ' and ' + optimizer_name + ' took ' + str(t_end-t_start) + ' seconds, which yields ' + str((t_end-t_start)/nb_runs) + ' seconds per run.'
print(to_print)







#############################################
# Semi Optimal sampling
#############################################

regressor_name = 'SemiOptimalRegressor'

optimizer_name = 'SGD'
list_time = []
t_start = time.time()
for i in range(nb_runs):
    t_start_run = time.time()
    regressor = SemiOptimalRegressor(lr, batch_size, optimizer_name, x, y, x_test, y_test)
    losses = []
    accuracy = []
    for j in range(nb_steps):
        regressor.learn()
        correct_on_test_set, loss_on_test_set = regressor.evaluate_regressor()
        losses.append(loss_on_test_set)
        accuracy.append(correct_on_test_set)
        if j % 50 == 0 :
            print(j)
    t_end_run = time.time()
    list_time.append(t_end_run - t_start_run)
    title1 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_loss_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title1, 'wb'))
    title2 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_acc_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title2, 'wb'))
t_end = time.time()
pickle.dump(list_time, open('results/' + regressor_name + '_' + optimizer_name + '_list_time', 'wb'))
to_print = 'For ' + str(nb_runs) + ' runs, optimization with ' + regressor_name + ' and ' + optimizer_name + ' took ' + str(t_end-t_start) + ' seconds, which yields ' + str((t_end-t_start)/nb_runs) + ' seconds per run.'
print(to_print)


optimizer_name = 'SGD_M'
list_time = []
t_start = time.time()
for i in range(nb_runs):
    t_start_run = time.time()
    regressor = SemiOptimalRegressor(lr, batch_size, optimizer_name, x, y, x_test, y_test)
    losses = []
    accuracy = []
    for j in range(nb_steps):
        regressor.learn()
        correct_on_test_set, loss_on_test_set = regressor.evaluate_regressor()
        losses.append(loss_on_test_set)
        accuracy.append(correct_on_test_set)
        if j % 50 == 0 :
            print(j)
    t_end_run = time.time()
    list_time.append(t_end_run - t_start_run)
    title1 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_loss_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title1, 'wb'))
    title2 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_acc_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title2, 'wb'))
t_end = time.time()
pickle.dump(list_time, open('results/' + regressor_name + '_' + optimizer_name + '_list_time', 'wb'))
to_print = 'For ' + str(nb_runs) + ' runs, optimization with ' + regressor_name + ' and ' + optimizer_name + ' took ' + str(t_end-t_start) + ' seconds, which yields ' + str((t_end-t_start)/nb_runs) + ' seconds per run.'
print(to_print)


optimizer_name = 'RMSP'
list_time = []
t_start = time.time()
for i in range(nb_runs):
    t_start_run = time.time()
    regressor = SemiOptimalRegressor(lr, batch_size, optimizer_name, x, y, x_test, y_test)
    losses = []
    accuracy = []
    for j in range(nb_steps):
        regressor.learn()
        correct_on_test_set, loss_on_test_set = regressor.evaluate_regressor()
        losses.append(loss_on_test_set)
        accuracy.append(correct_on_test_set)
        if j % 50 == 0 :
            print(j)
    t_end_run = time.time()
    list_time.append(t_end_run - t_start_run)
    title1 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_loss_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title1, 'wb'))
    title2 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_acc_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title2, 'wb'))
t_end = time.time()
pickle.dump(list_time, open('results/' + regressor_name + '_' + optimizer_name + '_list_time', 'wb'))
to_print = 'For ' + str(nb_runs) + ' runs, optimization with ' + regressor_name + ' and ' + optimizer_name + ' took ' + str(t_end-t_start) + ' seconds, which yields ' + str((t_end-t_start)/nb_runs) + ' seconds per run.'
print(to_print)



optimizer_name = 'ADAM'
list_time = []
t_start = time.time()
for i in range(nb_runs):
    t_start_run = time.time()
    regressor = SemiOptimalRegressor(lr, batch_size, optimizer_name, x, y, x_test, y_test)
    losses = []
    accuracy = []
    for j in range(nb_steps):
        regressor.learn()
        correct_on_test_set, loss_on_test_set = regressor.evaluate_regressor()
        losses.append(loss_on_test_set)
        accuracy.append(correct_on_test_set)
        if j % 50 == 0 :
            print(j)
    t_end_run = time.time()
    list_time.append(t_end_run - t_start_run)
    title1 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_loss_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title1, 'wb'))
    title2 = 'results/' + regressor_name + '_' + optimizer_name + '_lre2_MNIST_acc_' + str(nb_steps) + '_steps_' + str(i)
    pickle.dump(losses, open(title2, 'wb'))
t_end = time.time()
pickle.dump(list_time, open('results/' + regressor_name + '_' + optimizer_name + '_list_time', 'wb'))
to_print = 'For ' + str(nb_runs) + ' runs, optimization with ' + regressor_name + ' and ' + optimizer_name + ' took ' + str(t_end-t_start) + ' seconds, which yields ' + str((t_end-t_start)/nb_runs) + ' seconds per run.'
print(to_print)












