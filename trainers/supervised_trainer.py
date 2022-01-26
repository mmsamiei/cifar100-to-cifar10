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

        if 'scheduler' in kwargs:
            self.scheduler = kwargs['scheduler']
            self.has_scheduler = True
        else:
            self.has_scheduler = False
    
    def a_epoch(self, epoch_num):
        self.model.to(self.device)
        self.model = self.model.train()
        lossess = []
        pbar = tqdm(self.train_dataloader)
        for x,y in pbar:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            loss = supervised_loss_computation.supervised_loss_computation(self.model, self.head_num, x, y)
            loss.backward()
            self.optimizer.step()
            lossess.append(loss.item())
            pbar.set_description(f"Epoch {epoch_num}")
            pbar.set_postfix(loss = lossess[-1])
        mean_losses = sum(lossess)/len(lossess)
        pbar.set_postfix(loss = mean_losses)
        return {'epoch_loss_mean': mean_losses, 'epoch_lossess': lossess}
    
    def run(self, num_epoch=10):
        train_losses = []
        validation_accs = []
        for i in tqdm(range(num_epoch)):
            self.current_epoch_num += 1
            epoch_result = self.a_epoch(self.current_epoch_num)
            train_losses.append(epoch_result)
            if self.current_epoch_num % self.validation_period == 0:
                epoch_acc = tester.test(self.val_dataloader, self.model, self.head_num, 
                    self.device, tqdm_description=f"Epoch {self.current_epoch_num} Validation")
                validation_accs.append(epoch_acc)
                #print(f"Accuracy of Epoch {self.current_epoch_num} is {epoch_acc}:")
            
            if self.has_scheduler:
                self.scheduler.step()
        
        if validation_accs:
            print(f"max of accuracies is {max(validation_accs)} \n")
        




