from include import *
import MyModel
import MyDataset
import numpy

Device = "cuda" if torch.cuda.is_available() else "cpu"
#Device = "cpu"
print("Using {} device".format(Device))


def run_forward(G,input_img,size,save_path):
    in_img = cv.imread(input_img)
    in_img = cv.resize(in_img,dsize = size)
    tmp = numpy.zeros((1,3,in_img.shape[0],in_img.shape[1]))
    tmp[0] = numpy.transpose(in_img,(2,0,1))
    tmp = torch.from_numpy(tmp).to(Device)
    out_img = G(tmp)
    out_img = out_img[0].cpu().detach().numpy()
    out_img = numpy.transpose(out_img,(1,2,0))
    cv.imwrite(save_path,out_img)