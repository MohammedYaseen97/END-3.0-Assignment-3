# END-3.0-Assignment-3

**Data** <br />
The data has been extracted from [torchvision MNIST](https://pytorch.org/vision/stable/datasets.html#mnist) dataset. Along with a Dataset class, wherein we are adding a second input along with the MNIST image. To do this, we add a random int (between 1 and 9) as an input in the \__getitem\__ method, and add the sum of this random int and the mnist label as the second output.
```
#Train Dataset object
class TrainDataset(Dataset):
  def __init__(self):
    self.my_data = train_set
  
  def __getitem__(self, index):
    self.rand = random.randint(1, 9)
    self.inp_2 = torch.as_tensor(float(self.rand))
    self.img, self.label = self.my_data[index]
    self.summ = torch.as_tensor(self.label + self.rand)
    self.label = torch.as_tensor(self.label)
    return (self.img, torch.unsqueeze(self.inp_2, 0), torch.unsqueeze(self.label, 0), torch.unsqueeze(self.summ, 0))

  def __len__(self):
    return len(self.my_data)
```
<br /> <br />

**Architecture** <br />
We have 3 different neural networks.<br />
1. NetworkImg : Accepts a batch of images (tensor size : Batch size, 1, 28, 28) and outputs (tensor size : Batch size, 1, 12x8x8).<br />
2. NetworkInt : Accepts a batch of integers (tensor size : Batch size, 1, 1) and outputs (tensor size : Batch size, 1, 10). We then concatenate outputs of NetworkImg and NetworkInt to form (tensor size : Batch size, 1, 12x8x8+10). This combined input is passed onto the final layers (NetworkFinal).<br />
4. NetworkFinal : Accepts a batch of (tensor size : Batch size, 1, 12x8x8+10) and outputs (tensor size : Batch size, 1, 29).
```
outputImg = networkImg(images) # Pass Batch Images

outputInt = networkInt(rands) # Pass Batch Random No.s

finalInput = torch.cat((outputImg, outputInt), dim = 1) # Concatenate both Intermediate Outputs

outputs = networkFinal(finalInput) # Final Outputs
```
<br /> <br />

**Loss and Evaluation** <br />
We have divided each output (tensor size : 1 , 29) into 2 parts (tensor size : 1, 10) for the mnist label, and the other (tensor size : 1, 19) for the sum.<br />
We have applied __cross entropy__ loss separately on each part and summed the result to give final loss. The reason for doing that is : Cross Entropy Loss uses softmax, which makes sure only 1 class is highlighted in the result. We want only one class label to be true in 10, and we only need only one sum label to be true in 19. Cross entropy helps accomplish this task.<br />
Alternatively, we can evaluate on the whole tensor without dividing it into parts by using __binary cross entropy__ loss, which supports multi label classification, but it has other problems.
```
def custom_loss(outputs, labels, sums):

  loss1 = F.cross_entropy(torch.squeeze(outputs[:, :10]), torch.squeeze(labels)) # Cross entropy loss function for the mnist label
  loss2 = F.cross_entropy(torch.squeeze(outputs[:, 10:]), torch.squeeze(sums))  # Cross entropy loss function for the sum

  loss = loss1 + loss2

  return loss
```
<br /> <br />

**Results** <br />
Training on 10 epochs, we have the following results in training set:
```
Training set size :  50000
epoch 0 total_labels_correct: 46968 labels_accuracy: 93.94 total_sums_correct: 20904 sums_accuracy: 41.81 loss: 2555.1579919457436
epoch 1 total_labels_correct: 48136 labels_accuracy: 96.27 total_sums_correct: 31195 sums_accuracy: 62.39 loss: 1667.4679094552994
epoch 2 total_labels_correct: 48322 labels_accuracy: 96.64 total_sums_correct: 35562 sums_accuracy: 71.12 loss: 1383.4799114763737
epoch 3 total_labels_correct: 48339 labels_accuracy: 96.68 total_sums_correct: 38744 sums_accuracy: 77.49 loss: 1184.4521821141243
epoch 4 total_labels_correct: 48340 labels_accuracy: 96.68 total_sums_correct: 40249 sums_accuracy: 80.5 loss: 1098.0139260739088
epoch 5 total_labels_correct: 48436 labels_accuracy: 96.87 total_sums_correct: 41487 sums_accuracy: 82.97 loss: 1002.5089165568352
epoch 6 total_labels_correct: 48463 labels_accuracy: 96.93 total_sums_correct: 42407 sums_accuracy: 84.81 loss: 945.6608574390411
epoch 7 total_labels_correct: 48443 labels_accuracy: 96.89 total_sums_correct: 43129 sums_accuracy: 86.26 loss: 908.1297766044736
epoch 8 total_labels_correct: 48362 labels_accuracy: 96.72 total_sums_correct: 43396 sums_accuracy: 86.79 loss: 889.9875260666013
epoch 9 total_labels_correct: 48412 labels_accuracy: 96.82 total_sums_correct: 43587 sums_accuracy: 87.17 loss: 891.0338998809457
Time taken : 202.6733570098877 seconds
```
And the following results in validation set :
```
Validation set size :  10000
Validation results : 
total_labels_correct: 9679 labels_accuracy: 96.79 total_sums_correct: 8945 sums_accuracy: 89.45
```
<br /><br />

Group 21 members : <br />

Mohammed Yaseen (47.yaseen@gmail.com)<br />
Mayank Singhal (singhal.mayank77@gmail.com)<br />
Ravi Vaishnav (ravivaishnav20@gmail.com)<br />
Sundeep Joshi<br />



