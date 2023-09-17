#Load an image into an array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Waves import Waves
import math

class ImageUtils:
    def loadNorm(path, slicePx, threshold = None):
        img = tf.keras.preprocessing.image.load_img(path, color_mode='rgb')
        image_array  = tf.keras.utils.img_to_array(img)
        img = tf.image.rgb_to_grayscale(image_array)
        img = tf.image.resize(img, [slicePx,slicePx])
        target = tf.keras.utils.img_to_array(img)
        target = np.reshape(target, [slicePx,slicePx])
        target = target / np.max( target )
        if threshold != None:
            target[np.where(target < threshold)] = 0
            target[np.where(target >= threshold)] = 1
        return target
    
    def loadNormMultiple(paths, slicePx):
        imgs = []
        for path in paths:
            img = ImageUtils.loadNorm(path, slicePx)
            imgs.append(img)
        return imgs    
        
    def show_real_representation(elements_to_show, title = "3D representation of the model", legend = None, amp = None, save = False, folder_to_save = None, end_dec_plot = False, target = None, gif_plot = False, apperture = None, k = None, effect_of_one = False, gif_arrayAmp = False, mse = None):  
        # NOTE: In order to be able to move the plot, you must go to Tools -> Preferences -> IPython console -> Graphics and in Graphics backend change the backend from Inline(default) to Qt5, then after visualize you can change it to default again
        # NOTE: If you make more than plot they all overlap so just change to Qt5 if you are going to make just one plot
        fig = plt.figure()
        if gif_plot: 
            if effect_of_one:
                ax = fig.add_subplot(4, 1, 1, projection='3d')
            else:
                ax = fig.add_subplot(2, 1, 1, projection='3d')
        if gif_arrayAmp: ax = fig.add_subplot(2, 1, 1, projection='3d')
        
        else: ax = fig.add_subplot(projection='3d') 
        
        possible_colors = ["blue", "red", "green", "grey", "yellow"]
        
        elements_to_show = np.array(elements_to_show, dtype=object)
        for i in range(elements_to_show.shape[0]): 
            if amp != None and i == elements_to_show.shape[0]-1:
                ax.scatter(elements_to_show[i][:,0], elements_to_show[i][:,1], elements_to_show[i][:,2], 
                           c = amp, s = 3 * 256/elements_to_show[i].shape[0], cmap = 'gist_heat')
            else : 
                ax.scatter(elements_to_show[i][:,0], elements_to_show[i][:,1], elements_to_show[i][:,2], 
                           c = possible_colors[i % len(possible_colors)], s = 3 * 256/elements_to_show[i].shape[0])
            if end_dec_plot and i == 0:
                ax.scatter(elements_to_show[i][:,0], elements_to_show[i][:,1], elements_to_show[i][:,2], 
                           c = target, s = 3 * 256/elements_to_show[i].shape[0], cmap = 'gist_heat')
            # s control the size of the points in plot, so I choose 256/elements_to_show[i].shape[0] because is a way to make that the emmiters have s=1
            # and then I multiply by 3 so it doesn't look too small
        ax.set_title(title)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend(legend) if legend != None else None
        ax.view_init(azim=110) # If the init view point want to be changed
        
        if gif_plot:
            ax.set_zlim3d(0, 0.16) 
            if effect_of_one:
                ax = fig.add_subplot(4, 1, 2)
            else:
                ax = fig.add_subplot(2, 1, 2)
            eNormals = Waves.constNormals(elements_to_show[0], [0,0,1])
            props = Waves.calcPropagatorsPistonsToPoints(elements_to_show[0], eNormals, elements_to_show[1], k, apperture)
            
            ones = tf.complex(np.ones([1,elements_to_show[0].shape[0]]), np.zeros([1,elements_to_show[0].shape[0]]))
            result = ones @ props
            result_reshape = tf.reshape(result, [int(np.sqrt(props.shape[1])), int(np.sqrt(props.shape[1]))])
            
            plt.imshow(tf.abs(result_reshape), cmap='gist_heat')
            plt.colorbar()
            
            if effect_of_one:
                # Now I'm going to try to show the effect of just one
                just_one = np.zeros([1,elements_to_show[0].shape[0]])
                em_to_show = 6
                just_one[0, em_to_show] = 1
                
                ax = fig.add_subplot(4, 1, 3, projection='3d')
                ax.scatter(elements_to_show[0][[i for i,x in enumerate(elements_to_show[0]) if i!=em_to_show-1],0], elements_to_show[0][[i for i,x in enumerate(elements_to_show[0]) if i!=em_to_show-1],1], elements_to_show[0][[i for i,x in enumerate(elements_to_show[0]) if i!=em_to_show-1],2], 
                            c = 'blue', s = 3 * 256/elements_to_show[0].shape[0])
                ax.scatter(elements_to_show[0][em_to_show-1,0], elements_to_show[0][em_to_show-1,1], elements_to_show[0][em_to_show-1,2], 
                           c = 'yellow', s = 3 * 256/elements_to_show[0].shape[0])
                ax.scatter(elements_to_show[1][:,0], elements_to_show[1][:,1], elements_to_show[1][:,2], 
                           c = 'red', s = 3 * 256/elements_to_show[1].shape[0])
                ax.set_zlim3d(0, 0.16)
                ax.view_init(azim=110)
                
                ax = fig.add_subplot(4, 1, 4)
                just_one_c = tf.complex(just_one, np.zeros([1,elements_to_show[0].shape[0]]))
                result2 = just_one_c @ props
                result2_reshape = tf.reshape(result2, [int(np.sqrt(props.shape[1])), int(np.sqrt(props.shape[1]))])
                
                plt.imshow(tf.abs(result2_reshape), cmap='gist_heat')
                plt.colorbar()
            
        if gif_arrayAmp:
            ax.set_zlim3d(0, 1) 
            ax = fig.add_subplot(2, 1, 2)
            #plot amplitude field
            plt.imshow(amp, cmap = 'gist_heat')
            plt.colorbar()
            plt.title(("Log Loss:"+str(round(-math.log(mse),4))))
            plt.show()
            
        if save:
            plt.savefig("./PDF_figures/" + folder_to_save + "/" + title + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()
        return
    #TODO
    #def loadNormFromPath(dirPath, slicePx):
    