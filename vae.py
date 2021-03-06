import os,path,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time

class BatchGenerator:
    def __init__(self):
        self.folderPath = "sep"
        self.imagePath = glob.glob(self.folderPath+"/*/*.png")
        print "loaded %d images"%len(self.imagePath)
        self.imgSize = (64,64)
        assert self.imgSize[0]==self.imgSize[1]

    def getOne(self,idx):
        x   = np.zeros( (1,self.imgSize[0],self.imgSize[1],1), dtype=np.float32)

        img = cv2.imread(self.imagePath[idx],cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img,axis=2)
        dmin = min(img.shape[0],img.shape[1])
        img = img[int(0.5*(img.shape[0]-dmin)):int(0.5*(img.shape[0]+dmin)),int(0.5*(img.shape[1]-dmin)):int(0.5*(img.shape[1]+dmin)),:]
        img = cv2.resize(img,self.imgSize)
        img = np.expand_dims(img,axis=2)
        x[0,:,:,:] = img / 255.

        return x,None

    def getBatch(self,nBatch):
        x   = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],1), dtype=np.float32)
        for i in range(nBatch):
            img = cv2.imread(self.imagePath[random.randint(0,len(self.imagePath)-1)],cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img,axis=2)
            dmin = min(img.shape[0],img.shape[1])
            img = img[int(0.5*(img.shape[0]-dmin)):int(0.5*(img.shape[0]+dmin)),int(0.5*(img.shape[1]-dmin)):int(0.5*(img.shape[1]+dmin)),:]
            img = cv2.resize(img,self.imgSize)
            img = np.expand_dims(img,axis=2)
            x[i,:,:,:] = img / 255.

        return x,None

class VAE:
    def __init__(self,isTraining,imageSize,labelSize,args):
        self.nBatch = args.nBatch
        self.learnRate = args.learnRate
        self.zdim = args.zdim
        self.isTraining = isTraining
        self.imageSize = imageSize
        self.saveFolder = args.saveFolder
        self.reload = args.reload
        self.labelSize = labelSize
        self.initOP = None
        self.buildModel()

        return

    def _fc_variable(self, weight_shape,name="fc"):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            weight_shape    = (input_channels, output_channels)

            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer())
            bias   = tf.get_variable("b", [weight_shape[1]], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            input_channels  = int(weight_shape[2])
            output_channels = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape     , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [output_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _deconv_variable(self, weight_shape,name="conv"):
        with tf.variable_scope(name):
            # check weight_shape
            w = int(weight_shape[0])
            h = int(weight_shape[1])
            output_channels = int(weight_shape[2])
            input_channels  = int(weight_shape[3])
            weight_shape = (w,h,input_channels, output_channels)
            # define variables
            weight = tf.get_variable("w", weight_shape    , initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias   = tf.get_variable("b", [input_channels], initializer=tf.constant_initializer(0.0))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def _deconv2d(self, x, W, output_shape, stride=1):
        # x           : [nBatch, height, width, in_channels]
        # output_shape: [nBatch, height, width, out_channels]
        return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,stride,stride,1], padding = "SAME",data_format="NHWC")

    def leakyReLU(self,x,alpha=0.1):
        return tf.maximum(x*alpha,x) 

    def calcImageSize(self,dh,dw,stride):
        return int(math.ceil(float(dh)/float(stride))),int(math.ceil(float(dw)/float(stride)))

    def loadModel(self, model_path=None):
        if model_path: self.saver.restore(self.sess, model_path)

    def buildDecoder(self,z,label=None,reuse=False,isTraining=True):
        dim_0_h,dim_0_w = self.imageSize[0],self.imageSize[1]
        dim_1_h,dim_1_w = self.calcImageSize(dim_0_h, dim_0_w, stride=2)
        dim_2_h,dim_2_w = self.calcImageSize(dim_1_h, dim_1_w, stride=2)
        dim_3_h,dim_3_w = self.calcImageSize(dim_2_h, dim_2_w, stride=2)
        dim_4_h,dim_4_w = self.calcImageSize(dim_3_h, dim_3_w, stride=2)

        with tf.variable_scope("Decoder") as scope:
            if reuse: scope.reuse_variables()

            if label:
                l = tf.one_hot(label,self.labelSize,name="label_onehot")
                h = tf.concat([z,l],axis=1,name="concat_z")
                dim_next = self.zdim + self.labelSize
            else:
                h = z
                dim_next = self.zdim

            # fc1
            self.d_fc1_w, self.d_fc1_b = self._fc_variable([dim_next,dim_4_h*dim_4_w*128],name="fc1")
            h = tf.matmul(h, self.d_fc1_w) + self.d_fc1_b
            h = self.leakyReLU(h)
            #
            h = tf.reshape(h,(self.nBatch,dim_4_h,dim_4_h,128))

            # deconv4
            self.d_deconv4_w, self.d_deconv4_b = self._deconv_variable([5,5,128,64],name="deconv4")
            h = self._deconv2d(h,self.d_deconv4_w, output_shape=[self.nBatch,dim_3_h,dim_3_w,64], stride=2) + self.d_deconv4_b
            #h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNorm2")
            h = self.leakyReLU(h)

            # deconv3
            self.d_deconv3_w, self.d_deconv3_b = self._deconv_variable([5,5,64,32],name="deconv3")
            h = self._deconv2d(h,self.d_deconv3_w, output_shape=[self.nBatch,dim_2_h,dim_2_w,32], stride=2) + self.d_deconv3_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNorm1")
            h = self.leakyReLU(h)

            # deconv2
            self.d_deconv2_w, self.d_deconv2_b = self._deconv_variable([5,5,32,8],name="deconv2")
            h = self._deconv2d(h,self.d_deconv2_w, output_shape=[self.nBatch,dim_1_h,dim_1_w,8], stride=2) + self.d_deconv2_b
            #h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNorm2")
            h = self.leakyReLU(h)

            # deconv1
            self.d_deconv1_w, self.d_deconv1_b = self._deconv_variable([5,5,8,1],name="deconv1")
            h = self._deconv2d(h,self.d_deconv1_w, output_shape=[self.nBatch,dim_0_h,dim_0_w,1], stride=2) + self.d_deconv1_b

            # sigmoid
            y = tf.sigmoid(h)

            ### summary
            if reuse:
                tf.summary.histogram("d_fc1_w"   ,self.d_fc1_w)
                tf.summary.histogram("d_fc1_b"   ,self.d_fc1_b)
                tf.summary.histogram("d_deconv1_w"   ,self.d_deconv1_w)
                tf.summary.histogram("d_deconv1_b"   ,self.d_deconv1_b)
                tf.summary.histogram("d_deconv2_w"   ,self.d_deconv2_w)
                tf.summary.histogram("d_deconv2_b"   ,self.d_deconv2_b)
                tf.summary.histogram("d_deconv3_w"   ,self.d_deconv3_w)
                tf.summary.histogram("d_deconv3_b"   ,self.d_deconv3_b)
                tf.summary.histogram("d_deconv4_w"   ,self.d_deconv4_w)
                tf.summary.histogram("d_deconv4_b"   ,self.d_deconv4_b)

        return y

    def buildEncoder(self,y,isTraining=True,label=None,reuse=False):
        with tf.variable_scope("Encoder") as scope:
            if reuse: scope.reuse_variables()

            # conditional layer
            if label:
                l = tf.one_hot(label,self.labelSize,name="label_onehot")
                l = tf.reshape(l,[self.nBatch,1,1,self.labelSize])
                k = tf.ones([self.nBatch,self.imageSize[0],self.imageSize[1],self.labelSize])
                k = k * l
                h = tf.concat([y,k],axis=3)
                dim_next = 1+self.labelSize
            else:
                h = y
                dim_next = 1

            # conv1
            self.e_conv1_w, self.e_conv1_b = self._conv_variable([5,5,dim_next,8],name="econv1")
            h = self._conv2d(h,self.e_conv1_w, stride=2) + self.e_conv1_b
            #h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="dNorm1")
            h = self.leakyReLU(h)

            # conv2
            self.e_conv2_w, self.e_conv2_b = self._conv_variable([5,5,8,32],name="econv2")
            h = self._conv2d(h,self.e_conv2_w, stride=2) + self.e_conv2_b
            #h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="dNorm2")
            h = self.leakyReLU(h)

            # conv3
            self.e_conv3_w, self.e_conv3_b = self._conv_variable([5,5,32,64],name="econv3")
            h = self._conv2d(h,self.e_conv3_w, stride=2) + self.e_conv3_b
            h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="eNorm1")
            h = self.leakyReLU(h)

            # conv4
            self.e_conv4_w, self.e_conv4_b = self._conv_variable([5,5,64,128],name="econv4")
            h = self._conv2d(h,self.e_conv4_w, stride=2) + self.e_conv4_b
            #h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="dNorm2")
            h = self.leakyReLU(h)

            h_mu = h_sigma = h

            # fc_mu
            n_b, n_h, n_w, n_f = [int(x) for x in h_mu.get_shape()]
            h_mu = tf.reshape(h_mu,[self.nBatch,n_h*n_w*n_f])
            self.e_fc_mu_w, self.e_fc_mu_b = self._fc_variable([n_h*n_w*n_f,self.zdim],name="fc_mu")
            h_mu = tf.matmul(h_mu, self.e_fc_mu_w) + self.e_fc_mu_b

            # fc_sigma
            n_b, n_h, n_w, n_f = [int(x) for x in h_sigma.get_shape()]
            h_sigma = tf.reshape(h_sigma,[self.nBatch,n_h*n_w*n_f])
            self.e_fc_sigma_w, self.e_fc_sigma_b = self._fc_variable([n_h*n_w*n_f,self.zdim],name="fc_sigma")
            h_lnsigma = tf.matmul(h_sigma, self.e_fc_sigma_w) + self.e_fc_sigma_b

            ### summary
            if not reuse:
                tf.summary.histogram("e_conv1_w"   ,self.e_conv1_w)
                tf.summary.histogram("e_conv1_b"   ,self.e_conv1_b)
                tf.summary.histogram("e_conv2_w"   ,self.e_conv2_w)
                tf.summary.histogram("e_conv2_b"   ,self.e_conv2_b)
                tf.summary.histogram("e_conv3_w"   ,self.e_conv3_w)
                tf.summary.histogram("e_conv3_b"   ,self.e_conv3_b)
                tf.summary.histogram("e_conv4_w"   ,self.e_conv4_w)
                tf.summary.histogram("e_conv4_b"   ,self.e_conv4_b)
                tf.summary.histogram("e_fc_mu_w"   ,self.e_fc_mu_w)
                tf.summary.histogram("e_fc_mu_b"   ,self.e_fc_mu_b)
                tf.summary.histogram("e_fc_sigma_w"   ,self.e_fc_sigma_w)
                tf.summary.histogram("e_fc_sigma_b"   ,self.e_fc_sigma_b)

        return h_mu,h_lnsigma

    def buildModel(self):
        # define variables
        self.x      = tf.placeholder(tf.float32, [self.nBatch, self.imageSize[0], self.imageSize[1], 1],name="image")
        #self.l      = tf.placeholder(tf.int32  , [self.nBatch],name="label")

        self.z_mu, self.z_lnsigma = self.buildEncoder(self.x,isTraining=self.isTraining) # lnsigma = ln(sigma)... This admits to take -inf,+inf and make the calculation easier
        #self.z_mu      = tf.clip_by_value(self.z_mu,0.,1e+2)
        #self.z_lnsigma = tf.clip_by_value(self.z_lnsigma,0.,1e+2)
        # z -> [u_1,u_2,...],[s_1,s_2,...]
        rand        = tf.random_normal([self.nBatch,self.zdim]) # normal distribution
        self.z      = rand * tf.exp(self.z_lnsigma) + self.z_mu

        self.y      = self.buildDecoder(self.z,isTraining=self.isTraining)
        self.y_sample = self.buildDecoder(self.z,reuse=True,isTraining=False)

        self.gen_loss = - tf.reduce_sum( self.x * tf.log( tf.clip_by_value(self.y,1e-20,1e+20)) + (1.-self.x) * tf.log( tf.clip_by_value(1.-self.y,1e-20,1e+20))) # bbernoulli negative log likelihood. May be replaced by RMS?
        self.lat_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mu) + tf.exp(self.z_lnsigma)**2 - 2.*self.z_lnsigma - 1.)
        self.loss     = self.gen_loss + self.lat_loss

        # define optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learnRate).minimize(self.loss)

        ### summary
        tf.summary.scalar("gen_loss"      ,self.gen_loss)
        tf.summary.scalar("lat_loss"      ,self.lat_loss)
        tf.summary.scalar("loss"          ,self.loss    )
        tf.summary.histogram("z_mu"       ,self.z_mu   )
        tf.summary.histogram("z_sigma"    ,tf.exp(self.z_lnsigma))

        #############################
        # define session
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
        self.sess = tf.Session(config=config)

        #############################
        ### saver
        self.saver = tf.train.Saver(max_to_keep=0)
        self.summary = tf.summary.merge_all()
        if self.saveFolder: self.writer = tf.summary.FileWriter(self.saveFolder, self.sess.graph)

        #############################
        ### initializer
        self.initOP = tf.global_variables_initializer()
        self.sess.run(self.initOP)


        return

    def train(self,f_batch):

        def tileImage(imgs):
            d = int(math.sqrt(imgs.shape[0]-1))+1
            h = imgs[0].shape[0]
            w = imgs[0].shape[1]
            r = np.zeros((h*d,w*d,3),dtype=np.float32)
            for idx,img in enumerate(imgs):
                idx_y = int(idx/d)
                idx_x = idx-idx_y*d
                r[idx_y*h:(idx_y+1)*h,idx_x*w:(idx_x+1)*w,:] = img
            return r
        
        if self.saveFolder and not os.path.exists(os.path.join(self.saveFolder,"images")):
            os.makedirs(os.path.join(self.saveFolder,"images"))

        self.loadModel(self.reload)

        step = -1
        start = time.time()
        while True:
            step += 1

            batch_images,_ = f_batch(self.nBatch)
            _,loss,gen_loss,lat_loss,z,summary,y = self.sess.run([self.optimizer,self.loss,self.gen_loss,self.lat_loss,self.z,self.summary,self.y],feed_dict={self.x:batch_images})

            if step>0 and step%10==0:
                self.writer.add_summary(summary,step)

            if step%1000==0:
                print "%6d: loss=%.4e, loss(gen)=%.4e, loss(lat)=%.4e; time/step = %.2f sec"%(step,loss,gen_loss,lat_loss,time.time()-start)
                start = time.time()

                l0 = np.array([x%10 for x in range(self.nBatch)],dtype=np.int32)
                z1 = np.random.normal(0.,1.,[self.nBatch,self.zdim])

                g_image1 = self.sess.run(self.y_sample,feed_dict={self.z:z1})
                #g_image2 = self.sess.run(self.y_sample,feed_dict={self.z:z2,self.l:l0})
                cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_real.png"%step),tileImage(batch_images)*255.)
                cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_reco.png"%step),tileImage(y)*255.)
                cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_fake.png"%step),tileImage(g_image1)*255.)
                #cv2.imwrite(os.path.join(self.saveFolder,"images","img_%d_fake2.png"%step),tileImage(g_image2)*255.)
                self.saver.save(self.sess,os.path.join(self.saveFolder,"model.ckpt"),step)

    def test(self,z,loadModel=True):
        if self.saveFolder and not os.path.exists(os.path.join(self.saveFolder,"timages")):
            os.makedirs(os.path.join(self.saveFolder,"timages"))

        y = self.sess.run(self.y,feed_dict={self.z:z})
        return y[0]*255.
