model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
