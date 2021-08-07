model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

for epoch in range(1, epochs + 1):
    # Train Mode
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # backpropagation 계산하기 전에 0으로 기울기 계산
        output = model(data)
        loss = F.nll_loss(output, target)  # https://pytorch.org/docs/stable/nn.html#nll-loss
        loss.backward()  # 계산한 기울기를 
        optimizer.step()  

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
