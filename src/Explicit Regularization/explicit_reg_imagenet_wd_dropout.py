import os
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
  
def create_train_dataset(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    
    train_transforms = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        train_transforms
    )
    
    return train_dataset
  
def create_val_dataset(data_dir):
    val_dir = os.path.join(data_dir, 'val')
    
    val_transforms = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(
        val_dir,
        val_transforms
    )
    
    return val_dataset
  
def create_data_loaders(data_dir, batch_size=32, workers=4, pin_memory=True):
    train_dataset = create_train_dataset(data_dir)
    val_dataset = create_val_dataset(data_dir)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    ) 
    return train_loader, val_loader

train_loader, val_loader = create_data_loaders('/Users/shubham/Documents/Rutgers University/MS in Data Science/Spring 2024/Financial Data Mining/Project/')


def get_device():
  if torch.cuda.is_available():
      return torch.device('cuda')
  else:
      return torch.device('cpu')
device = get_device()

def evaluate_model(model, train_loader, test_loader):
  model.eval()
  correct_train = 0
  total_train = 0
  for i, data in enumerate(test_loader, 0):
    images, labels = data[0].to(device), data[1].to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, dim=1)
    total_train += labels.size(0)
    correct_train += (preds == labels).sum().item()
  train_accuracy = (correct_train / total_train)

  correct_test = 0
  total_test = 0            
  for i, data in enumerate(train_loader, 0):
    images, labels = data[0].to(device), data[1].to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, dim=1)
    total_test += labels.size(0)
    correct_test += (preds == labels).sum().item()
  test_accuracy = (correct_test / total_test)
  return train_accuracy, test_accuracy

def fit(epochs, model, train_loader, test_loader, optimizer, scheduler, start_epoch=0, step_count=0):
  history_train = []
  history_test = []
  criterion = nn.CrossEntropyLoss().to(device)
  model.train()
  for epoch in range(start_epoch,epochs):  
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)
      optimizer.zero_grad()
      outputs, aux_outputs = model(inputs)
      loss1 = criterion(outputs, labels)
      loss2 = criterion(aux_outputs, labels)
      loss = loss1 + 0.4*loss2
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      step_count += 1
      if step_count%10000==0:
        train_acc, test_acc = evaluate_model(model, train_loader, test_loader)
        model.train()
        history_train.append(train_acc)
        history_test.append(test_acc)
        log_entry = str(step_count)+","+str(train_acc)+","+str(test_acc)+"\n"
        with open("model_1_new.log", "a") as f:
          f.write(log_entry)
        print("Epoch: {} | Step: {} | loss: {:.4f} | Train acc: {:.4f} | Val acc: {:.4f}".format(epoch+1,step_count, running_loss,train_acc, test_acc))
    scheduler.step()
    filename = '/Users/shubham/Documents/Rutgers University/MS in Data Science/Spring 2024/Financial Data Mining/Project/'
    torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step_count': step_count
            },filename)
    print("----Checkpoint Saved----",epoch+1) 
  return model,history_train,history_test

if not os.path.exists('/Users/shubham/Documents/Rutgers University/MS in Data Science/Spring 2024/Financial Data Mining/Project/'):
  os.makedirs('/Users/shubham/Documents/Rutgers University/MS in Data Science/Spring 2024/Financial Data Mining/Project/')

net = models.inception_v3(pretrained=False).to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.95)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
if resume_training == True:
  checkpoint = torch.load('/Users/shubham/Documents/Rutgers University/MS in Data Science/Spring 2024/Financial Data Mining/Project/.pth.tar')
  net.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  start_epoch = checkpoint['epoch']
  step_count = checkpoint['step_count']
else:
  start_epoch = 0
  step_count = 0
trained_model,history_train,history_test = fit(100, net, train_loader, val_loader, optimizer, scheduler, start_epoch=start_epoch, step_count=step_count)