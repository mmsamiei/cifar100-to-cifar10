from computations import supervised_loss_computation
from tqdm.auto import tqdm
from . import tester

class SupervisedJointTrainer():

    def __init__(self, train_dataloaders: list, val_dataloaders: list, model,\
         optimizer, device, head_nums, **kwargs) -> None:
        self.train_dataloaders = train_dataloaders
        self.number_tasks = len(self.train_dataloaders)
        self.val_dataloaders = val_dataloaders
        self.model = model
        self.optimizer = optimizer 
        self.device = device
        self.head_num = 0
        self.current_epoch_num = 0
        self.validation_period = 5
        self.head_nums = head_nums

        if 'task_names' in kwargs:
            self.task_names = kwargs['task_names']
        else:
            self.task_names = [f'Task {i}' for i in range(self.number_tasks)]

        if 'scheduler' in kwargs:
            self.scheduler = kwargs['scheduler']
            self.has_scheduler = True
        else:
            self.has_scheduler = False
    
    def a_epoch(self, epoch_num):
        self.model.to(self.device)
        self.model = self.model.train()
        epoch_result = {}
        for i in range(self.number_tasks):
            lossess = []
            pbar = tqdm(self.train_dataloaders[i])
            for x,y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                loss = supervised_loss_computation.supervised_loss_computation(self.model, self.head_nums[i], x, y)
                loss.backward()
                self.optimizer.step()
                lossess.append(loss.item())
                task_name = self.task_names[i]
                pbar.set_description(f"{task_name} Epoch {epoch_num}")
                pbar.set_postfix(loss = lossess[-1])
            mean_losses = sum(lossess)/len(lossess)
            pbar.set_postfix(loss = mean_losses)
            epoch_result[str(task_name)] = {'epoch_loss_mean': mean_losses, 'epoch_lossess': lossess}
        return epoch_result
    
    def run(self, num_epoch=10):
        train_losses = []
        validation_accs = {}

        for i in range(self.number_tasks):
            validation_accs[self.task_names[i]] = []

        for i in tqdm(range(num_epoch)):
            self.current_epoch_num += 1
            epoch_result = self.a_epoch(self.current_epoch_num)
            train_losses.append(epoch_result)
            if self.current_epoch_num % self.validation_period == 0:
                for task_num in range(self.number_tasks):
                    task_name = self.task_names[task_num]
                    epoch_acc = tester.test(self.val_dataloaders[task_num], self.model,
                        self.head_nums[task_num], self.device, 
                        tqdm_description=f"{task_name} Epoch {self.current_epoch_num} Validation")
                    validation_accs[task_name].append(epoch_acc)
                #print(f"Accuracy of Epoch {self.current_epoch_num} is {epoch_acc}:")
            
            if self.has_scheduler:
                self.scheduler.step()
        for i in range(self.number_tasks):
            task_name = self.task_names[i]
            validation_acc = validation_accs[task_name]
            if validation_acc:
                print(f"{task_name} max of accuracies is {max(validation_acc)} \n")
        




