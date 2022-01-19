from computations import accuracy_computation
from tqdm.auto import tqdm

def test(test_dataloader, model, head_num, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    for x,y in tqdm(test_dataloader):
        x = x.to(device)
        y = y.to(device)
        temp = accuracy_computation.accuracy_computation(model, head_num, x, y)
        correct += temp['num_correct']
        total += temp['total']
    return correct/total

