from computations import supervised_loss_computation

class SupervisedTrainer():

    def __init__(self, train_dataloader, val_dataloader, model,\
         optimizer, num_epochs, **kwargs) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.optimizer = optimizer 
        self.num_epochs = num_epochs
        
        self.head_num = 0
        if 'head_num' in kwargs:
            self.head_num = kwargs['head_num']
    
    def a_epoch(self):
        for x,y in self.train_dataloader:
            self.optimizer.zero_grad()
            loss = supervised_loss_computation.supervised_loss_computation(self.model, self.head_num, x, y)
            loss.backward()
            self.optimizer.step()

