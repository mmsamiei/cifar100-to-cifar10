from computations import accuracy_computation
from tqdm.auto import tqdm

def test(test_dataloader, model, head_num, device, tqdm_description=""):
    model = model.eval()
    model.to(device)
    correct = 0
    total = 0
    pbar = tqdm(test_dataloader)
    pbar.set_description(tqdm_description)
    for x,y in pbar:
        x = x.to(device)
        y = y.to(device)
        temp = accuracy_computation.accuracy_computation(model, head_num, x, y)
        correct += temp['num_correct']
        total += temp['total']
        pbar.set_postfix(acc = correct/total)
    return correct/total

