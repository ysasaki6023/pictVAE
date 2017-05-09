import os,path,sys,shutil
import numpy as np
import argparse
import matplotlib.pyplot as plt
from vae import BatchGenerator,VAE

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload","-l",dest="reload",type=str,required=True)
    parser.add_argument("--learnRate","-r",dest="learnRate",type=float,default=2e-3)
    parser.add_argument("--saveFolder","-s",dest="saveFolder",type=str,default="models")
    parser.add_argument("--imgIndex","-i",dest="imgIndex",type=int,default=None)
    args = parser.parse_args()
    args.zdim = 100
    args.nBatch = 1
    vae = VAE(isTraining=False,imageSize=[64,64],labelSize=0,args=args)

    assert args.reload
    vae.loadModel(args.reload)

    if not args.imgIndex:
        z = np.random.normal(0.,1.,[1,args.zdim])
    else:
        b = BatchGenerator()
        f,_ = b.getOne(args.imgIndex)
        z = vae.sess.run(vae.z_mu,feed_dict={vae.x:f})
        print z
    y = vae.test(z)
    #print y
    y = np.tile(y,(1,1,3))
    plt.imshow(y)
    #plt.imshow(np.tile(f[0],(1,1,3)))
    plt.show()
