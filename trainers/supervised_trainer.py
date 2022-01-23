from computations import supervised_loss_computation
from tqdm.auto import tqdm
from . import tester

class SupervisedTrainer():

    def __init__(self, train_dataloader, val_dataloader, model,\
         optimizer, device, **kwargs) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.optimizer = optimizer 
        self.device = device
        self.head_num = 0
        self.current_epoch_num = 0
        self.validation_period = 5
        if 'head_num' in kwargs:
            self.head_num = kwargs['head_num']
    
    def a_epoch(self):
        self.model.to(self.device)
        self.model = self.model.train()
        lossess = []
        for x,y in tqdm(self.train_dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            loss = supervised_loss_computation.supervised_loss_computation(self.model, self.head_num, x, y)
            loss.backward()
            self.optimizer.step()
            lossess.append(loss.item())
        return {'epoch_loss_mean': sum(lossess)/len(lossess), 'epoch_lossess': lossess}
    
    def run(self, num_epoch=10):
        train_losses = []
        validation_accs = []
        for i in tqdm(range(num_epoch)):
            self.current_epoch_num += 1
            print(f"Epoch {self.current_epoch_num}: ")
            epoch_result = self.a_epoch()
            print(f"Mean Loss of Epoch {self.current_epoch_num} is {epoch_result['epoch_loss_mean']}:")
            train_losses.append(epoch_result)
            if self.current_epoch_num % self.validation_period == 0:
                epoch_acc = tester.test(self.val_dataloader, self.model, self.head_num, self.device)
                validation_accs.append(epoch_acc)
                print(f"Accuracy of Epoch {self.current_epoch_num} is {epoch_acc}:")
        
        if validation_accs:
            print(f"max of accuracies is {max(validation_accs)}")
        




