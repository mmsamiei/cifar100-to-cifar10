from computations import supervised_loss_computation

class SupervisedTrainer():

    def __init__(self, train_dataloader, val_dataloader, model,\
         optimizer, **kwargs) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.optimizer = optimizer 
        
        self.head_num = 0
        if 'head_num' in kwargs:
            self.head_num = kwargs['head_num']
    
    def a_epoch(self):
        lossess = []
        for x,y in self.train_dataloader:
            self.optimizer.zero_grad()
            loss = supervised_loss_computation.supervised_loss_computation(self.model, self.head_num, x, y)
            loss.backward()
            self.optimizer.step()
            lossess.append(loss.item())
        return {'epoch_loss_mean': sum(lossess)/len(lossess),\
             'epoch_loss_max': max(lossess), 'epoch_loss_min': min(lossess)}
    
    def run(self, num_epoch=10):
        for epoch_num in range(num_epoch):
            print(f"Epoch {epoch_num}: ")
            epoch_result = self.a_epoch()
            print(f"Mean Loss of Epoch {epoch_num} is {epoch_result['epoch_loss_mean']}:")


