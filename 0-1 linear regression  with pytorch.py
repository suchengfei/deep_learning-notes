#从0-1 pytorch实现线性回归
import torch 
import matplotlib.pyplot as plt
x = torch.randn(100,1)*10
y = 2*x+1 +torch.randn(100,1)*2
plt.scatter(x,y)
plt.show()

w =torch.randn(1,requires_grad = True)
b = torch.randn(1,requires_grad = True)

def forword(x):
    return x*w+b
def loss_fn(y_pred,y_true):
    return((y_pred-y_true)**2).mean()

lr = 0.01
epochs = 1000
for epoch in range(epochs):
    y_pred = forword(x)
    loss = loss_fn(y_pred,y)
    loss.backward()

    with torch.no_grad():
        w -=lr*w.grad
        b -=lr*b.grad
        w.grad.zero_()  
        b.grad.zero_()
    
    if (epoch+1)%20 ==0:
        print(f"epoch[{epoch+1}/{epochs}],Loss:{loss.item():.4f},w:{w.item():.4f},b:{b.item():.4f}")

