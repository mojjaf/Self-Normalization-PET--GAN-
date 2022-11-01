
import tensorflow as tf
from mainargs import get_args

args = get_args()
INPUT_CHANNELS=args.input_channel
OUTPUT_CHANNELS = args.output_channel

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result



def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, INPUT_CHANNELS])
    filter=64

    down_stack = [
    downsample(filter, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(filter*2, 4),  # (batch_size, 64, 64, 128)
    downsample(filter*4, 4),  # (batch_size, 32, 32, 256)
    downsample(filter*8, 4),  # (batch_size, 16, 16, 512)
    downsample(filter*8, 4),  # (batch_size, 8, 8, 512)
    downsample(filter*8, 4),  # (batch_size, 4, 4, 512)
    downsample(filter*8, 4),  # (batch_size, 2, 2, 512)
    downsample(filter*8, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
    upsample(filter*8, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(filter*8, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(filter*8, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(filter*8, 4),  # (batch_size, 16, 16, 1024)
    upsample(filter*4, 4),  # (batch_size, 32, 32, 512)
    upsample(filter*2, 4),  # (batch_size, 64, 64, 256)
    upsample(filter, 4),  # (batch_size, 128, 128, 128)
    ]
    
  
    

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

     # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def Generator_ResUnet():
    inputs = tf.keras.layers.Input(shape=[256, 256, INPUT_CHANNELS])

    down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]
    
  
    
    filters=[512,512,512,512,256,128,64]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down,filt in zip (down_stack,filters):
        x = down(x)
        x1= residual_block(x, [filt], 4, strides=[1,1])

        skips.append(x1)
        
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip, filt in zip(up_stack, skips, filters):
        
        x = up(x)
        x= tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)



def Generator_Attention():
    inputs = tf.keras.layers.Input(shape=[256, 256, INPUT_CHANNELS])

    down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]
    
  
    
    filters=[512,512,512,512,256,128,64]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down,filt in zip (down_stack,filters):
        x = down(x)
        skips.append(x)
        
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip, filt in zip(up_stack, skips, filters):
        
        x = up(x)
        net= attention(x, skip, filt, 2)
        x= tf.keras.layers.Concatenate()([x, net])
        x = self_attention(x)

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, INPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, OUTPUT_CHANNELS], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
    kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)





def residual_block(x, num_filters, kernel_size, strides):
    """Residual Unet block layer
    Consists of batch norm and relu, folowed by conv, batch norm and relu and 
    final convolution. The input is then put through 
    Args:
        x: tensor, image or image activation
        num_filters: list, contains the number of filters for each subblock
        kernel_size: int, size of the convolutional kernel
        strides: list, contains the stride for each subblock convolution
        name: name of the layer
    Returns:
        x1: tensor, output from residual connection of x and x1
    """

    if len(num_filters) == 1:
        num_filters = [num_filters[0], num_filters[0]]

    x1 = tf.keras.layers.BatchNormalization()(x)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Conv2D(filters=num_filters[0], 
                                kernel_size=kernel_size, 
                                strides=strides[0], 
                                padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Conv2D(filters=num_filters[1], 
                                kernel_size=kernel_size,
                                strides=strides[1], 
                                padding='same')(x1)

    x = tf.keras.layers.Conv2D(filters=num_filters[-1],
                                    kernel_size=1,
                                    strides=strides[0],
                                    padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
                                                            
    x_out = tf.keras.layers.Add()([x, x1])

    return x_out


def attention(tensor, att_tensor, filters, size):
    initializer = tf.random_normal_initializer(0., 0.02)

    g1 = tf.keras.layers.Conv2D(filters, kernel_size=size,strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False)(tensor)
    x1 = tf.keras.layers.Conv2D(filters, kernel_size=size,strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False)(att_tensor)
    net = tf.keras.layers.add([g1, x1])
    net = tf.keras.layers.Activation('relu')(net)
    net = tf.keras.layers.Conv2D(1, kernel_size=size,strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False)(net)
    net = tf.nn.sigmoid(net)
    #net = tf.keras.layers.Concatenate()([att_tensor, net])
    net = net * att_tensor
    return net

def self_attention(att_tensor):
    C=int(att_tensor.shape[3])
    H=int(att_tensor.shape[2])
    W=int(att_tensor.shape[1])
    
    initializer = tf.random_normal_initializer(0., 0.02)
    ################################# First Block
    g1 = tf.keras.layers.Conv2D(1, kernel_size=1,strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False)(att_tensor)
    g1= tf.keras.layers.Reshape((1, 1, H*W), input_shape=g1.shape)(g1)
    g1=tf.keras.layers.Activation('softmax')(g1)
    
    x1 = tf.keras.layers.Conv2D(C/2, kernel_size=1,strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False)(att_tensor)
    x1= tf.keras.layers.Reshape((H*W,int(C/2) ), input_shape=x1.shape)(x1) 
    g1=tf.squeeze(g1,axis=1)
    
    net1=tf.matmul(g1,x1)
    
    net1=tf.expand_dims(net1, axis=1)
    net1=tf.transpose(net1,perm=[0,1, 2, 3])
    
    ################################# Second Block
    net1 = tf.keras.layers.Conv2D(C, kernel_size=1,strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False)(net1)

    net1 = tf.keras.layers.BatchNormalization()(net1)
    net1 = tf.nn.sigmoid(net1)
    
    net1= tf.keras.layers.Reshape((C,1,1),input_shape=net1.shape)(net1)
    net1=tf.transpose(net1,perm=[0,2,3,1])
    net1=tf.multiply(att_tensor, net1)
        
    g2 = tf.keras.layers.Conv2D(int(C/2), kernel_size=1,strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False)(net1)
    
    g2 = tf.keras.layers.GlobalMaxPool2D()(g2)
    g2= tf.keras.layers.Reshape((1, int(C/2)), input_shape=g2.shape)(g2)
    g2=tf.keras.layers.Activation('softmax')(g2)
    
    x2 = tf.keras.layers.Conv2D(int(C/2), kernel_size=1,strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False)(net1)
    x2= tf.keras.layers.Reshape((int(C/2),H*W), input_shape=x2.shape)(x2)
    
    net2=tf.matmul(x2, g2, transpose_a=True,transpose_b=True)
   
    net2= tf.keras.layers.Reshape((1,H, W), input_shape=net2.shape)(net2)
    net2 = tf.nn.sigmoid(net2)
    net2=tf.transpose(net2,perm=[0,2,3,1])
   
    net= tf.multiply(net2 , net1)
    
    return net
