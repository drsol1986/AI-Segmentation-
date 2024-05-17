import tensorflow as tf
#import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tf
import numpy as np

from tqdm import tqdm
from colorama import Fore
import random
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

#Custom implementation of a hybrid densenet/3D U-Net: Orginal Verison of 3D U-net: https://arxiv.org/abs/1606.06650; Densenet: https://arxiv.org/abs/1608.06993

#No data augmentation added at this time; will do so later  :(

##############################
#function to decipher tfrecord for training and augementation
def parser(tfrecord):

    features = tf.parse_single_example(tfrecord,{'test/image': tf.FixedLenFeature([], tf.string),
               'test/label': tf.FixedLenFeature([], tf.string),
               'test/depth': tf.FixedLenFeature([], tf.int64),
               'test/height': tf.FixedLenFeature([], tf.int64),
               'test/width': tf.FixedLenFeature([], tf.int64)})
    height = tf.cast(features["test/height"], tf.int32)
    width = tf.cast(features["test/width"], tf.int32)
    depth = tf.cast(features["test/depth"], tf.int32)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['test/image'], tf.float32) #data type has to be the same as original image

    label = tf.decode_raw(features['test/label'], tf.float32) #data type has to be the same as original label

    # Reshape image data into the original shape
    image = tf.reshape(image, [height, width,depth])
    label = tf.reshape(label, [height, width,depth])

    #Cast input to float32 and label to int32
    label = tf.cast(label,tf.int32)
    image = tf.cast(image,tf.float32)

    #Crop image for input size if desired
    #image = tf.transpose(image,perm=[2, 0, 1, 3])
    #image = tf.image.resize_image_with_crop_or_pad(image, 256, 224)
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 192)
    #image = tf.transpose(image,perm=[1, 2, 0, 3])

    #label = tf.transpose(label,perm=[2, 0, 1, 3])
    #label = tf.image.resize_image_with_crop_or_pad(label, 256, 224)
    label = tf.image.resize_image_with_crop_or_pad(label, 224, 192)
    #label = tf.transpose(label,perm=[1, 2, 0, 3])

    image = (image - tf.reduce_min(image))/(tf.reduce_max(image) - tf.reduce_min(image)) #Normalize input image
    image = image/tf.reduce_max(image)

    label = tf.one_hot(label, depth=2,on_value=1,off_value=0,axis=-1 ) #Encode label data to one_hot array for softmax cross entropy (depth = number of classes; on_value = foreground; off_value = background)

    return image, label
################################

################################
#Encoding layer: Differences from Dense-Unet -- uses growth of 12 (ie number of channels at each convolution layer) and iterates a number times and concatenates each feature map
#Currently pool size and kernal size are each set to [2x2x1] so that all features maps are halved at the height and width but no the slice number--this allows for variable input sizes
# example: x_con.shape = [1,H,W,S]; after max-pooling with pool_size = [2x2x1], kernal = [2x2x1] --> x_con.shape = [1, H/2, W/2, S] 
def encoder_layer(x_con, iterations, name,training, pool=True):
    
    with tf.name_scope("encoder_block_{}".format(name)):
        for i in range(iterations):
            
            x = tf.layers.conv3d(x_con,12,kernel_size=[3,3,3],padding='SAME')
            #x = tf.layers.dropout(x,rate=0.1,training=training)
            x = tf.layers.batch_normalization(x,training=training)
            x = tf.nn.relu(x)
            x_con = tf.concat([x,x_con], axis = 4)
        if pool is False:
            return x_con
        
        #x = tf.layers.conv3d(x_con,12,kernel_size=[1,1,1],padding='SAME')
        #x = tf.layers.dropout(x,rate=0.1,training=training)
        #x = tf.layers.batch_normalization(x,training=training,renorm=True)
        #x = tf.nn.relu(x)
        pool = tf.layers.max_pooling3d(x_con,pool_size = [2,2,1], strides=[2,2,1],data_format='channels_last')

        return x_con, pool
#################################

#################################
#Decoding layer: Same as traditional 3D Unet--upsamples total number of features except in slice directions
def decoder_layer(input_, x, ch, name):
        

    up = tf.layers.conv3d_transpose(input_,filters=12,kernel_size = [2,2,1],strides = [2,2,1],padding='SAME',name='upsample'+str(name), use_bias=False)
    up = tf.concat([up,x], axis=-1, name='merge'+str(name))
    return up
#################################


    
###########################################
#Dense-Unet model:  -- uses a dice loss function along w/ softmax cross entropy;  
class DUnet():
    def __init__(self, x, training):
        #self.filters = filters
        self.training = training
        self.model = self.DU_net(x)

    
    def DU_net(self,input_):
#        with tf.device('/GPU:1'):


        conv1, pool1 = encoder_layer(input_,iterations=2,name="encode_"+str(1),training=self.training, pool=True)
        conv2, pool2 = encoder_layer(pool1,iterations=4,name="encode_"+str(2),training=self.training, pool=True)
        conv3, pool3 = encoder_layer(pool2,iterations=6,name="encode_"+str(3),training=self.training, pool=True)
        conv4, pool4 = encoder_layer(pool3,iterations=8,name="encode_"+str(4),training=self.training, pool=True)
        conv5, pool5 = encoder_layer(pool4,iterations=10,name="encode_"+str(5),training=self.training, pool=True)
        conv6 = encoder_layer(pool5,iterations=12,name="encode_"+str(6),training=self.training, pool=False)
        up1 = decoder_layer(conv6,conv5,10,name=1)
        conv7 = encoder_layer(up1,iterations=10,name="conv"+str(6),training=self.training, pool=False)
        up2 = decoder_layer(conv7,conv4,8,name=2)
        conv8 = encoder_layer(up2,iterations=8,name="encode_"+str(7),training=self.training, pool=False)
        up3 = decoder_layer(conv8,conv3,6,name=3)
        conv9 = encoder_layer(up3,iterations=6,name="encode_"+str(8),training=self.training, pool=False)
        up4 = decoder_layer(conv9,conv2,4,name=4)
        conv10 = encoder_layer(up4,iterations= 4,name="encode_"+str(9),training=self.training, pool=False)
        up5 = decoder_layer(conv10,conv1,2,name=5)
        conv11 = encoder_layer(up5,iterations= 2,name="encode_"+str(10),training=self.training, pool=False)

        #The final layer to generate a probability map from the CNN: channels input should be total number of classes (ie. a binary segmentation of 0 or 1 has a total number of classes = 2)
        score = tf.layers.conv3d(conv11,2,(1,1,1),name='logits',padding='SAME')

        return score

#Dice loss function: used to force the network towards better segmentation accuracy in small number of training datasets w/ high background to foreground ratios.
def cost_dice(logits, labels,name='cost'):
    with tf.name_scope('cost'):
        eps = 1e-5
        logits = tf.nn.softmax(logits)
        logits1 = tf.argmax(logits, axis=4) #Reduce dimensions of CNN output
        #logits2 = logits[...,2]>0.5
        #logits1 = logits[...,1]>0.5

        #logits2 = tf.cast(logits2,tf.float32)
        logits1 = tf.cast(logits1,tf.float32)

        #labels = tf.argmax(labels, axis=4) #Reduce dimensions of labels to original mask
        #labels2 = labels[...,2]
        labels1 = labels[...,1]

        #labels2 = tf.cast(labels2,tf.float32)
        labels1 = tf.cast(labels1,tf.float32)

        #log2 = tf.reshape(logits2,[1,-1]) #reshape to linearize -- if increasing batch size; the dimensions needs to be changed but not sure how (maybe, batch_size=2; tf.reshape(logits, [2,-1]))
        log1 = tf.reshape(logits1,[1,-1]) #reshape to linearize -- if increasing batch size; the dimensions needs to be changed but not sure how (maybe, batch_size=2; tf.reshape(logits, [2,-1]))
        
        #labels2 = tf.reshape(labels2,[1,-1]) #reshape to linearize -- if increasing batch size; the dimensions needs to be changed but not sure how
        labels1 = tf.reshape(labels1,[1,-1]) #reshape to linearize -- if increasing batch size; the dimensions needs to be changed but not sure how
        
        #inte2 = tf.multiply(log2,labels2)
        inte1 = tf.multiply(log1,labels1)

        #inter2 = eps + tf.reduce_sum(inte2) #Total intersection b/w CNN output and label
        inter1 = eps + tf.reduce_sum(inte1) #Total intersection b/w CNN output and label

        #union2 =  tf.reduce_sum(log2) + tf.reduce_sum(labels2)+eps #The total union of CNN output and labels
        union1 =  tf.reduce_sum(log1) + tf.reduce_sum(labels1)+eps #The total union of CNN output and labels

        loss = (1-tf.reduce_mean(2* inter1/ (union1)))# + (1-tf.reduce_mean(2* inter1/ (union1))) #The dice loss function = 1 - dice score, as such w/ the loss decreasing over iterations --> dice score increasing
        
        #For multi-channel segmentation, seperate dice losses were caulcated for each chennel and summed together as the total loss (maybe try averaging the dice losses to see if it improves)
        return loss

            
            
    
########################################
#Training Portion and Dataset Fetching

def Unet_train():
    #image_batch_placeholder = tf.placeholder("float32", shape=[1, 256,224,None, 1]) #By default: the image sizes are cropped to [128x96xS] (S = slice number) --> can change if desired but adjust parser function as well
    image_batch_placeholder = tf.placeholder("float32", shape=[1, 224,192,None, 1])
    #label_batch_placeholder = tf.placeholder(tf.float32, shape=[1, 256,224,None,2])
    label_batch_placeholder = tf.placeholder(tf.float32, shape=[1, 224,192,None,2])

    #vectorize label pixels for softmax cross entropy
    labels_pixels = tf.reshape(label_batch_placeholder, [-1, 2])
    
    training_flag = tf.placeholder(tf.bool)

    logits = DUnet(x = image_batch_placeholder, training=training_flag).model #Runs CNN, logits = raw CNN output

    #vectorize CNN ouput for softmax cross entropy
    logit = tf.reshape(logits,(-1, 2))
    


    #softmax cross entropy loss -- Difference from 3D U-Net: no weighted softmax cross entropy: was found to be unnecessary w/ dice loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_pixels, logits=logit))
    

    #Dice loss function
    dice_loss = cost_dice(logits=logits, labels = label_batch_placeholder)
    
    total_loss = dice_loss + loss
    

    tf.summary.scalar('total_loss', total_loss)

    #Track the total number of iterations: epoch* total number of iterations during one epoch
    global_step = tf.Variable(0, name='global_step', trainable=False)

    #Used a constant learning rate
    learning_rate = 0.0001
    tf.summary.scalar('learning_rate', learning_rate)

    #Used an Adam Optimizer, which seems to work well for small datasets trained for a short period of time
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(total_loss, global_step=global_step)


    summary_op = tf.summary.merge_all()  # merge all summaries into a single "operation" which we can execute in a session

    saver = tf.train.Saver(max_to_keep=5)
    all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])



    config = tf.ConfigProto(log_device_placement=False)
    #config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

#############################################
    #To Fetch and queue training dataset
    dataset = tf.data.TFRecordDataset('D:/ICAD_Projekt/AI_Segmentation/Training_setting1/setting1/train_new_134cases_run3.tfrecords') #location of tfrecord for training
    dataset = dataset.map(map_func=parser, num_parallel_calls=3)
    dataset = dataset.batch(1) #batch-size for training
    dataset = dataset.shuffle(buffer_size=50) #buffer to randomize dataset
    dataset = dataset.prefetch(50) #queue set-up
    dataset = dataset.repeat(450) 
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
##########################################


    summary_writer = tf.summary.FileWriter("./log", sess.graph) #location to save tensorboard log

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


    checkpoint = tf.train.get_checkpoint_state("./IC_seg_new") # Where to load the trained weights; loads the most recent weights in checkpoint txt file
    if(checkpoint != None):
        tf.logging.info("Restoring full model from checkpoint file %s",checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

    coord = tf.train.Coordinator()

    total = 134 # PW changed from 70 to 134 Total Number of training datasets/ Number of examples per batch (By default batch-size is set to 1, so total = Total Number of training datasets)
    check_points = 800
    for epoch in range(300): #Number of epochs to train for

       #Iterating over the total dataset
        for check_point in tqdm(range(134),    
        bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Fore.RESET)):

            image_out, truth = sess.run(next_element)
            image_out = np.expand_dims(image_out,axis=4) #Add grayscale value, input.shape = [1,128,96,S,1]
            _, training_loss, other_loss, _global_step, summary = sess.run([train_step, total_loss, dice_loss, global_step, summary_op],
                                             feed_dict={image_batch_placeholder: image_out,
                                                        label_batch_placeholder: truth,
                                                        training_flag: True})

            #plt.imshow(image_out[0,:,:,33,0])
            #plt.show()
            #plt.imshow(truth[0,:,:,33,1])
            #plt.show()

            if(bool(check_point%round(57) == 0) & bool(check_point != 0)):
                print(_)
                #The current global step: number of iterations * epoch
                print("global step: ", _global_step)

                #The total loss at this iteration
                print("training loss: ", training_loss)

                #The dice loss at this iteration
                print("other_loss:", other_loss)

                #The total number of parameters in CNN
                print(sess.run(all_trainable_vars))

                #Summary for tensorboard
                summary_writer.add_summary(summary, _global_step)


        # Where to save the trained weights for next epoch: should be the same as the checkpoint load path
        saver.save(sess, "./IC_seg_new/hb.ckpt", _global_step) 
        
        
        
        


    coord.request_stop()
    #coord.join(threads)
    sess.close()
    return 1



def main():
    tf.reset_default_graph()
    Unet_train()



if __name__ == '__main__':
    main()
