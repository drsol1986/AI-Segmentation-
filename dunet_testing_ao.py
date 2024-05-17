import time
s_t = time.time()
from dense_unet_ao import DUnet
import tensorflow as tf
#import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tf
import numpy as np
import os
from tqdm import tqdm
import scipy.io as io
import statistics
import argparse
import matplotlib.pyplot as plt
#from skimage import segmentation
#from skimage.feature import peak_local_max
#from skimage import morphology

#parser = argparse.ArgumentParser()
#parser.add_argument("--path")
#args = parser.parse_args()
#path = args.path

def feed_data():
    data_path = 'D:/ICAD_Projekt/AI_Segmentation/Training_setting1/IC_venous1_new.tfrecords'  # address to save the hdf5 file
    feature = {'test/image': tf.FixedLenFeature([], tf.string),
               'test/label': tf.FixedLenFeature([], tf.string),
               #'test/phases': tf.FixedLenFeature([], tf.int64),
               'test/depth': tf.FixedLenFeature([], tf.int64),
               'test/height': tf.FixedLenFeature([], tf.int64),
               'test/width': tf.FixedLenFeature([], tf.int64)}
    
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path])
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    height = tf.cast(features["test/height"], tf.int32)
    width = tf.cast(features["test/width"], tf.int32)
    depth = tf.cast(features["test/depth"], tf.int32)
    #phases = tf.cast(features["test/phases"], tf.int32)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['test/image'], tf.float32)
    
    # Cast label data into data type
    label = tf.decode_raw(features['test/label'], tf.float32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [height, width, depth])
    label = tf.reshape(label, [height, width, depth])
    #image = image[10:138,:,:]
    #label = label[10:138,:,:]
    #image = tf.transpose(image,perm=[2, 0, 1, 3])
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 192)
    #image = tf.transpose(image,perm=[1, 2, 0, 3])

    #label = tf.transpose(label,perm=[2, 0, 1, 3])
    label = tf.image.resize_image_with_crop_or_pad(label, 224, 192)
    #label = tf.transpose(label,perm=[1, 2, 0, 3])

    label = tf.cast(label,tf.float32)
    image = tf.cast(image,tf.float32)
    

    image = tf.expand_dims(image,axis = 0)
    label = tf.expand_dims(label,axis = 0)
    image = tf.expand_dims(image,axis = 4)
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, depth=2,on_value=1,off_value=0,axis=-1 )
    label = tf.cast(label,tf.float32)

    image = (image - tf.reduce_min(image))/(tf.reduce_max(image) - tf.reduce_min(image))
    image = image/tf.reduce_max(image)

    #label = tf.expand_dims(label,axis = 4)
    q = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.float32])
    enqueue_op = q.enqueue_many([image,label])
    #image, label = q.dequeue()
    qr = tf.train.QueueRunner(q,[enqueue_op])
    return image, label #phases
def cost_dice(logits, labels,name='cost'):
    with tf.name_scope('cost'):
        eps = 1e-5
        logits = tf.nn.softmax(logits)
        #logits = tf.argmax(logits, axis=4) #Reduce dimensions of CNN output
        #logits2 = logits[...,2]>0.5
        logits1 = logits[...,1]>0.5

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

        #dice2 = (tf.reduce_mean(2* inter2/ (union2)))
        dice1 = (tf.reduce_mean(2* inter1/ (union1))) #The dice loss function = 1 - dice score, as such w/ the loss decreasing over iterations --> dice score increasing
        #loss = (1-tf.reduce_mean(2* inter1/ (union1)))
        return dice1


def Unet_test():

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[None, 224,192,None, 1])
    label_batch_placeholder = tf.placeholder(tf.float32, shape=[None, 224, 192,None,2])
    labels_pixels = tf.reshape(label_batch_placeholder, [-1, 1])    #   if_training_placeholder = tf.placeholder(tf.bool, shape=[])
    training_flag = tf.placeholder(tf.bool)
    image_batch,label_batch = feed_data()


    #label_batch_dense = tf.arg_max(label_batch, dimension = 1)

 #   if_training = tf.Variable(False, name='if_training', trainable=False)

    logits = DUnet(x = image_batch_placeholder, training=training_flag).model
    #logits = logits>1
    d1 = cost_dice(logits,label_batch_placeholder)
    llogits = tf.nn.softmax(logits)
    
    

    checkpoint = tf.train.get_checkpoint_state("D:/ICAD_Projekt/AI_Segmentation/scripts/IC_script_and_data/weights_134cases_run3") #locatin of weights
    saver = tf.train.Saver()
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

 #  logits_batch = tf.to_int64(tf.arg_max(logits, dimension = 1))
    d = 0

    config = tf.ConfigProto(log_device_placement=False)
    all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])

    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tf.logging.info("Restoring full model from checkpoint file %s",checkpoint.model_checkpoint_path)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        accuracy_accu = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)
        d = []
        dd = []
        tru = []
        use = []
        gtruth = []
        low = []
        low_true = []
        low_loss = []
        num = []
        exe_time = []
        n =2

        for i in tqdm(range(int(n))):

            start_t = time.time()


            image_out, truth = sess.run([image_batch, label_batch])


            
            _, dice, llogit = sess.run([logits, d1, llogits], feed_dict={image_batch_placeholder: image_out,
                                                                                    label_batch_placeholder: truth,
                                                                                    training_flag: True})
            #infer_out = infer_out / np.amax(infer_out)
            #llogits = tf.nn.softmax(infer_out)
            unique = np.unique(llogit, return_counts=True)
            #print(llogit.shape)
            llogit = np.squeeze(llogit)
            #infer_out = np.argmax(llogit, axis=3)
            llogit1 = llogit[...,1]>0.5
            #llogit2 = llogit[...,2]>0.5
            llogit1 = llogit1*1
            #llogit2 = llogit2*2
            infer_out = llogit1
            #infer_out[infer_out==3] = 1
            truth = np.squeeze(truth)
            truth = np.argmax(truth, axis=3)
            #print(infer_out.shape)
            #print(truth.shape)
            #truth = truth >= 1
            
            #o = np.squeeze(infer_out)
            #t = np.squeeze(truth)
            #H,W = o.shape
            #o = np.reshape(o,(1,H*W))
            #t = np.reshape(t,(1,H*W))
            #inter = np.logical_and(o, t)
            #union = np.logical_or(o, t)
            #los = (np.sum(inter) + 1e-5)/(np.sum(union) + 1e-5)
            #infer_out = int(infer_out == 'true')
            #loss = cost_dice(infer_out,truth)
            #accuracy_out = np.asarray(accuracy_out)
            #print("label:", truth)
            #print("infer: ", guess)
            #print(' ')
            #loss = -(2*inte/unio)

            #print(Counter(truth))
            #print(ys.shape)
            #print(infer_out.shape)
            #unique = np.unique(infer_out, return_counts=True)
            #print(unique)

            #accuracy_accu = accuracy_out + accuracy_accu
            #O = np.float32(image_out[:,:,:,15,:])
          
            #Y = np.float32(truth[:,:,:,15])
            #print(Y.dtype)

            #lo = np.squeeze(soft[...,0])
            #print(logis.shape)
            data = np.squeeze(infer_out)
            truth = np.squeeze(truth)
            print('Dice score 1: ', dice)
            #print('Dice score 2: ', dice_2)
            d.append(dice)
            #dd.append(dice_2)
            print(data.shape)
            #loss = 1-loss
            """
            if loss <0.90 or loss2<0.90:
                print(dice)
                print(dice_2)
                print(i)
                low_d = data.transpose()
                low.append(low_d)
                low_t = truth.transpose()
                low_true.append(low_t)
                low_loss.append(loss)
                num.append(i)
            ave.append(loss)
            """
            #print(ys.shape)
           
            #gt = np.squeeze(Y)
           # if i > 300:


            mask = data.transpose()
            print(np.amax(mask))
            
            #mask = mask>1
            #from scipy import ndimage
            #distance = ndimage.distance_transform_edt(mask)
            #local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3,3)), labels=mask)
            #markers = morphology.label(local_maxi)
            #markers[~mask] = -1
            #mask = segmentation.random_walker(mask, markers)
            gtuse = truth.transpose()
            gtruth.append(gtuse)
            use.append(mask)
            fini_t = time.time()
            diff_t = fini_t - start_t
            exe_time.append(diff_t)
            #gt = gt.transpose()
            
               
            
            #G = (infer_out[:,:,:,15])
            #G = np.reshape(G,(256,256))
            #U = np.squeeze(Y)
            #W = np.squeeze(G)
            #lo = np.squeeze(O)
            #ys = (ys)
            #print(W.shape)
            #plt.imshow(data[...,25])
            #plt.show()
            #plt.pause(0.1)
            #plt.imshow(truth[...,25])
            #plt.show()
            #plt.pause(0.1)
            
            #plt.imshow(lo)
            #plt.show()


        """
            if len(a) > 1:
                gus = sorted(a,key=a.get)[-2]
                l, count = a.most_common(2)

                if count[1]/48 < 0.45:
                    gus = max(a,key=a.get)
            else:
                gus = max(a,key=a.get)
        """
                
            #if count / 48 >= .55
            #print(ans)
            #print(gus)
            
            

            
 #           I = image_batch
#            L = label_batch
            #Ai = tf.convert_to_tensor(image_batch[check_point], dtype=tf.float64)
            #Al= tf.convert_to_tensor(label_batch[check_point], dtype=tf.int32)
            #I = tf.expand_dims(Ai, axis = 0)
             
            #Al = tf.one_hot(Al, depth=7, on_value=1.0, off_value=0.0)
            #L = tf.expand_dims(Al, axis = 0)



        #print("patch accuracy:",accuracy_accu)
       # print("image accuracy:",d/n)
        print(sess.run(all_trainable_vars))
        io.savemat('./mask_venous1_new.mat',{'data':use})
        #io.savemat('/home/haben/Documents/MATLAB/low.mat',{'data':low})
        #io.savemat('/home/haben/Documevnts/MATLAB/low_truth.mat',{'truth':low_true})
        #io.savemat('/home/haben/Documents/MATLAB/low_num.mat',{'low_num':num})





        print("mean Ao: ", sum(d)/len(d))
        #print(sum(low_loss)/len(low_loss))

        #print(statistics.stdev(d))
        print("median Ao: ", statistics.median(d))

        #print("mean PA: ", sum(dd)/len(dd))
        #print(sum(low_loss)/len(low_loss))

        #print(statistics.stdev(dd))
        #print("median PA: ", statistics.median(dd))
        #print(sum(exe_time)/len(exe_time))
        #print(statistics.stdev(exe_time))
        #print(statistics.median(exe_time))
        #f_t = time.time()
        #print(s_t - f_t)

        io.savemat('./gt_all8_2b.mat',{'truth':gtruth})
        
        #print(len(ma))
        print(len(tru))
        
        

        #print(statistics.stdev(ave))


        

        tf.train.write_graph(sess.graph_def, 'graph/', 'my_graph.pb', as_text=False)

        coord.request_stop()
        coord.join(threads)
        sess.close()
    return 0



def main():
    tf.reset_default_graph()

    Unet_test()



if __name__ == '__main__':
    main()
