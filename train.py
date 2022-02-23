from torch import Tensor
from include import *
import MyModel
import MyDataset
import itertools
import numpy
import run_forward
from torch.autograd import Variable

Device = "cuda" if torch.cuda.is_available() else "cpu"
#Device = "cpu"
print("Using {} device".format(Device))

shape = (3,128,128)
lr = float(0.0002)
b1 = float(0.5)
b2 = float(0.999)

G_x2y = MyModel.Generator(shape,6).to(Device)
G_y2x = MyModel.Generator(shape,6).to(Device)
D_x = MyModel.Discriminator(shape).to(Device)
D_y = MyModel.Discriminator(shape).to(Device)

waifu_path = "D:\\Downloads\\anime_face"#input("动画人脸 : ")#Y
human_path = "D:\\Downloads\\seeprettyface_race_yellow"#input("真人人脸 : ")#X
dataset = MyDataset.imgs(waifu_path,human_path,(128,128))

torch.cuda.empty_cache()
optimizer_G = torch.optim.Adam(itertools.chain(G_x2y.parameters(), G_y2x.parameters()),
                               lr=lr, betas=(b1, b2))
optimizer_DX = torch.optim.Adam(D_x.parameters(), lr=lr, betas=(b1, b2))
optimizer_DY = torch.optim.Adam(D_y.parameters(), lr=lr, betas=(b1, b2))

modul_save_path = "D:\\Documents\\CycleGANdata\\exp1"#input("模型保存地址 : ")
testimg_path = "D:\\Documents\\CycleGANdata\\exp1\\Gimgs"#input("测试图片输出 : ")
test_input = "D:\\Documents\\CycleGANdata\\testin.jpg"#input("测试输入图片 : ")
Batch_size = 64
dataLoader = DataLoader(dataset,Batch_size,shuffle = True)

loss_fn_GAN = torch.nn.MSELoss().to(Device)
loss_fn_cycle = torch.nn.L1Loss().to(Device)
loss_fn_identity = torch.nn.L1Loss().to(Device)
Epach = int(1000)

for i in range(1,Epach+1):
    print("Epach : " + str(i))
    for T,(X,Y) in enumerate(dataLoader):
        X,Y = X.to(Device),Y.to(Device)
        valid = Variable(Tensor(numpy.ones(Batch_size)),requires_grad = False)
        fake = Variable(Tensor(numpy.zeros(Batch_size)),requires_grad = False)
        #----------
        #train G
        #----------
        G_x2y.train()
        G_y2x.train()
        optimizer_G.zero_grad()

        loss_id_x = loss_fn_identity(G_y2x(X),X)
        loss_id_y = loss_fn_identity(G_x2y(Y),Y)
        loss_id = (loss_id_x + loss_id_y)/2

        fake_Y = G_x2y(X)
        loss_GAN_x2y = loss_fn_GAN(D_y(fake_Y),valid)
        fake_X = G_y2x(Y)
        loss_GAN_y2x = loss_fn_GAN(D_x(fake_X),valid)
        loss_GAN = (loss_GAN_x2y + loss_GAN_y2x)/2

        recov_X = G_y2x(fake_Y)
        loss_cycle_xyx = loss_fn_cycle(recov_X,X)
        recov_Y = G_x2y(fake_X)
        loss_cycle_yxy = loss_fn_cycle(recov_Y,Y)
        loss_cycle = (loss_cycle_xyx + loss_cycle_yxy)/2

        total_loss_G = loss_GAN + float(10)*loss_cycle + float(5)*loss_id
        total_loss_G.backward()
        optimizer_G.step()
        print("    Batch : " + str(T) + "  G_loss : " + total_loss_G.detach().item())
        #----------
        #train D_X
        #---------
        optimizer_DX.zero_grad()
        loss_real = loss_fn_GAN(D_x(X),valid)
        loss_fake = loss_fn_GAN(D_x(fake_X),fake)
        loss_D_X = (loss_real + loss_fake)/2
        loss_D_X.backward()
        optimizer_DX.step()
        print("    Batch : " + str(T) + "  Dx_loss : " + loss_D_X.detach().item())
        #----------
        #train D_Y
        #---------
        optimizer_DY.zero_grad()
        loss_real = loss_fn_GAN(D_y(Y),valid)
        loss_fake = loss_fn_GAN(D_y(fake_Y),fake)
        loss_D_Y = (loss_real + loss_fake)/2
        loss_D_Y.backward()
        optimizer_DY.step()
        print("    Batch : " + str(T) + "  Dy_loss : " + loss_D_Y.detach().item())
    if i % 5 == 0:
        torch.save(G_x2y,modul_save_path + "\\" + "G_x2y_" + str(i))
        torch.save(G_y2x,modul_save_path + "\\" + "G_y2x_" + str(i))
        torch.save(D_x,modul_save_path + "\\" + "D_x_" + str(i))
        torch.save(D_y,modul_save_path + "\\" + "D_y_" + str(i))
    run_forward.run_forward(G_x2y,test_input,shape,testimg_path)
    
