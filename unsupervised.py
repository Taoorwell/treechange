import os
import torch
from dataloader import dataloader, get_device
from unets import ResUnet, DiceLoss
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Step 1: Datasets and Dataloader preparation
dataloader = dataloader(image_dir=r'images/', batch_size=1)

# Step 2: Model preparation
model = ResUnet(n_bands=7, n_classes=50)

# Step 3: Training configuration
epochs = 1
lr = 0.0001
iteration = 20
# get device
device = get_device()
# # loss function and optimizer defining
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# # To Do: cross entropy or dice loss
loss_function = torch.nn.CrossEntropyLoss()
dice_loss = DiceLoss()

# # model to device
model.to(device)

# model training loop with epochs
for epoch in range(epochs):
    # training on single loop with the whole dataloader
    epoch_sample_loss = []
    for batch_sample in dataloader:
        # training on single batch (image)
        batch_sample_loss = []
        for i in range(iteration):
            # repeat on single batch (image) with iteration times
            optimizer.zero_grad()
            model.train()
            image = batch_sample['Image'].to(device)
            segment = batch_sample['Segment'].to(device)
            # forward
            output = model(image)[0]
            # print(f'Output shape: {output.shape}')
            output_label = torch.argmax(output, dim=0)
            # print(f'Output label shape: {output_label.shape}')
            n_unique_labels = torch.unique(output_label)
            # print(f'Epoch:{epoch}, iteration:{i}, The number of unique labels: {len(n_unique_labels)}')

            # spatial refinements using segment in batch sample
            for i_segment in torch.unique(segment[0]):
                i_segment_labels = output_label[segment[0] == i_segment]
                u_i_segment_labels, hist = torch.unique(i_segment_labels, return_counts=True)
                output_label[segment[0] == i_segment] = u_i_segment_labels[torch.argmax(hist)]
            # loss and backpropagation
            # output = output.permute(1, 2, 0).contiguous().view(-1, 50)
            # output_label = output_label.view(-1)
            # loss = loss_function(output, output_label)
            loss = dice_loss(output, output_label)
            loss.backward()
            optimizer.step()
            batch_sample_loss.append(loss.item())
            # print(f'Epoch: {epoch}, iteration:{i}, loss:{loss.item()}')
            if len(n_unique_labels) <= 3:
                break
        print(f'Epoch: {epoch}, iteraion: {i}, the number of unique labels: {len(n_unique_labels)}, '
              f'single batch mean loss: {np.mean(batch_sample_loss)}')
        epoch_sample_loss.append(np.mean(batch_sample_loss))
    print(f'Epoch:{epoch}, epoch mean loss: {np.mean(epoch_sample_loss)}')

# model saved
torch.save(model.state_dict(), r'checkpoints/unsupervised_model')








