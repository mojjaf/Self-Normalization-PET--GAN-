#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf



print("\n * START CONFIGURATION* \n")

############################## Define the number of Physical and Virtual GPUs #############

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[5:6], 'GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=20000),
             tf.config.LogicalDeviceConfiguration(memory_limit=20000)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
        print(e)


print("\n * GPU Setup compelted... * \n")


################################# load libraries #######################################
import os 
import numpy as np 
import datetime
import time
from IPython import display
from utils import resize, load_image_data_from_directory, load_h5,normalize_tensor, load_files_from_directory
from models import Generator_Attention, Discriminator
from evaluation import evaluate_model
from losses import generator_loss, discriminator_loss
from mainargs import get_args


############################## Basic configurations ##################################
dataset= "geomcorrected"
mode='triple'  #2.5 D or single (2D)
experiment = "/experiments/Pix2Pix_SelfAtten_"+dataset+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/"  # 
model_name = "model_25D_"+dataset+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/" 
print(f"\nExperiment: {experiment}\n")
args = get_args()
project_dir = args.main_dir
#data_dir=args.data_dir

experiment_dir = project_dir+experiment

if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir) 
    
models_dir = experiment_dir+model_name
if not os.path.exists(models_dir):
    os.makedirs(models_dir) 
    
output_preds_2D = experiment_dir+'pix2pix2.5D_predictions/'

if not os.path.exists(output_preds_2D):
    os.makedirs(output_preds_2D)



############################## LOAD DATA AND PREPROCESS #############################
PATH=args.data_dir
IMG_WIDTH = args.image_size
IMG_HEIGHT = args.image_size
INPUT_CHANNELS=args.input_channel
OUTPUT_CHANNELS = args.output_channel
print(INPUT_CHANNELS, OUTPUT_CHANNELS)
train_dirNorm=os.path.join(PATH, 'target','train')
train_dirUnnorm=os.path.join(PATH,'input',dataset, mode, 'train')
#val_dirNorm=os.path.join(PATH,dataset, 'A','val')
#val_dirUnnorm=os.path.join(PATH,dataset, 'B','val')
test_dirNorm=os.path.join(PATH, 'target','val')
test_dirUnnorm=os.path.join(PATH,'input',dataset, mode, 'val')

start_time = time.time()
print("\n * Loading data... (this might take a few minutes) * \n")

with tf.device('/device:cpu:0'):
    input_image,real_image=load_files_from_directory(train_dirUnnorm,train_dirNorm,IMG_HEIGHT, IMG_WIDTH)
    print("\n * Trainset: DONE * \n")
    #input_image_val,real_image_val=load_image_data_from_directory(val_dirUnnorm,val_dirNorm,IMG_HEIGHT, IMG_WIDTH)
    #print("\n * Validationset: DONE * \n")
    input_image_test,real_image_test=load_files_from_directory(test_dirUnnorm,test_dirNorm,IMG_HEIGHT, IMG_WIDTH)
    print("\n * Testset:DONE * \n")

print("Total number training samples ", len(input_image))
#print("Total number validation samples ",len(input_image_val))
print("Total number test samples ",len(input_image_test))

print("\n * All data samples successfully loaded. * \n")

# In[19]:
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
])



print("\n * Creating Zipped Tensors... * \n")
BUFFER_SIZE = len(input_image)
BATCH_SIZE = args.batch_size


augment=False
AUTOTUNE = tf.data.AUTOTUNE

with tf.device('/device:cpu:0'):
    datasetx = tf.data.Dataset.from_tensor_slices((input_image))

    datasety = tf.data.Dataset.from_tensor_slices((real_image))
    train_dataset = tf.data.Dataset.zip((datasetx, datasety))

    if augment:
        train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
        #train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE) # Use buffered prefetching on all datasets.
    
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)

print("\n * Training dataset successfully created. * \n")


# In[20]:


#with tf.device('/device:cpu:0'):
 #   datasetxVal = tf.data.Dataset.from_tensor_slices((input_image_val))
 #   datasetyVal = tf.data.Dataset.from_tensor_slices((real_image_val))
 #   val_dataset = tf.data.Dataset.zip((datasetxVal, datasetyVal)).batch(BATCH_SIZE)

#print("\n * Validation dataset successfully created. * \n")

# In[21]:


with tf.device('/device:cpu:0'):
    datasetxTest = tf.data.Dataset.from_tensor_slices((input_image_test))
    datasetyTest = tf.data.Dataset.from_tensor_slices((real_image_test))
    test_dataset = tf.data.Dataset.zip((datasetxTest, datasetyTest)).batch(BATCH_SIZE)

print("\n * Test dataset successfully created. * \n")
print("\n * Preprocessing: DONE * \n")
print("--- total processing time: %s seconds ---" % (time.time() - start_time))

#############################################################################################




# In[9]:
########################### GAN MODELS configuration ########
generator = Generator_Attention()
#generator.summary()

discriminator = Discriminator()
#discriminator.summary()

learning_rate=args.lr

generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)



############################# TRAINING Configuration ########


generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)

@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
       
print("\n * GAN models successfully configured... * \n")

# In[31]:

import datetime
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))



def fit(train_ds, steps):
    #example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    epoch=0

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            display.clear_output(wait=True)
            #print("\n * TRAINING... * \n")

            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            #generate_images(generator, example_input, example_target)
            print("\n * TRAINING... * \n")
            print(f"Step: {step//1000}k")

        train_step(input_image, target, step)
        
        # Training step
        if (step+1) % 10 == 0:
            print('.', end='', flush=True)


        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        # Save the model every 20k steps
        if (step + 1) % 20000 == 0:
            epoch+=1
            generator.save_weights('./gen_'+ str(epoch) + '.h5')
            print("\n * MODEL SAVED * \n")

    

#################################### Start Training 
print("\n * Training Started ... * \n")
os.chdir(experiment_dir)
step_size =args.steps
    
fit(train_dataset, steps=step_size)  #100000 or 52 epochs




print("\n * SAVING PREDICTIONS... * \n")

for file_name in (os.listdir(test_dirUnnorm)):
    #print(file_name)
    real_A= load_h5(os.path.join(test_dirUnnorm, file_name))
    real_B= load_h5(os.path.join(test_dirNorm, file_name))
    
    real_A=np.array(real_A)
    real_B=np.array(real_B)

    if OUTPUT_CHANNELS ==1:
        real_B=np.expand_dims(real_B, axis=-1)

    if INPUT_CHANNELS==1:
        real_A=np.expand_dims(real_A, axis=-1)

    real_A,real_B=resize(real_A,real_B, IMG_HEIGHT, IMG_WIDTH)
    real_A,real_B=normalize_tensor(real_A,real_B)
    
    #outpath = r'/home/mojjaf/pix2pix_self_norm/test_results_fullDS_w3CE_bs1_40ep_doubleattention_ResUnet_100622/' 
    outpath=output_preds_2D
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    fake_B = generator(np.expand_dims(real_A, axis=0), training=True)  
    proto_tensor_A = tf.make_tensor_proto(real_A)
    real_A=tf.make_ndarray(proto_tensor_A)
    proto_tensor_B = tf.make_tensor_proto(real_B)
    real_B=tf.make_ndarray(proto_tensor_B)
    proto_tensor_fB = tf.make_tensor_proto(fake_B)
    fake_B=tf.make_ndarray(proto_tensor_fB)
    fake_B=np.squeeze(fake_B, axis=0)
    real_A = (255*(real_A - np.min(real_A))/np.ptp(real_A)).astype(int) 
    real_B = (255*(real_B - np.min(real_B))/np.ptp(real_B)).astype(int) 
    fake_B = (255*(fake_B - np.min(fake_B))/np.ptp(fake_B)).astype(int) 
    #plt.figure()
    #plt.imshow(real_A* 0.5 + 0.5)
    np.save(outpath+ file_name[:-4]+'_real_A.npy', real_A)
    np.save(outpath+ file_name[:-4]+'_real_B.npy', real_B)
    np.save(outpath+ file_name[:-4]+'_fake_B.npy', fake_B)


# In[ ]:
print("\n *  Saving DONE... * \n")

print("\n * CALCULATING EVALUATION METRICS... * \n")
Qreport=evaluate_model(generator,test_dataset, experiment_dir)

snr_cum=Qreport['SNRpr']### predicted vs high dose PET
ssi_cum=np.array(Qreport['SSIpr'])
mse_cum=np.array(Qreport['MSEpr'])


########## Mean values of the PSNR, SSIM, and MSE #########
snr_resPR_mean = np.mean(snr_cum)
ssim_resPR_mean = np.mean(ssi_cum)
mse_resPR_mean = np.mean(mse_cum)


########## Standard deviation values of the PSNR, SSIM, MSE#########
snr_resPR_std = np.std(snr_cum)
ssim_resPR_std = np.std(ssi_cum)
mse_resPR_std = np.std(mse_cum)

print("Mean of the PSNR is ", snr_resPR_mean, "+/-", snr_resPR_std) 
print("Mean of the SSIm is ", ssim_resPR_mean,"+/-",ssim_resPR_std) 
print("Mean of the MSE is",mse_resPR_mean,"+/-",mse_resPR_std)

Qreport.to_csv(experiment_dir+'results.csv', index=False)

print("\n * Inference mode successfully COMPLETED... * \n")
