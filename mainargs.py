import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_dir", type=str,default='/home/mojjaf/pix2pix_self_norm/Pix2Pix-Tensorflow-SelfNormalization/Pix2Pix_V.04',
                        help="project directory")
    parser.add_argument("--data_dir", type=str,default='/home/mojjaf/pix2pix_self_norm/Pix2Pix-Tensorflow-SelfNormalization/data/updatad_data/self_norm_dataset_v3/',
                        help="all image dir")
    parser.add_argument("--image_size", type=int, default=256,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="batch size")
    parser.add_argument("--output_channel", type=int, default=1,
                        help="output image dimension 2D or 2.5D")
    parser.add_argument("--input_channel", type=int, default=3,
                        help="input image dimension 2D or 2.5D")
    parser.add_argument("--nb_steps", type=int, default=40,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=35000,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default=None,
                        help="loss; mean_squared_error', 'mae', or 'l0' is expected")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default=r'/scratch/project_2003990/MinMotion_PET_CSC',
                        help="checkpoint dir")
    parser.add_argument("--model_path", type=str, default=r'/scratch/project_2003990/MinMotion_PET_CSC',
                        help="checkpoint dir")
   
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    return args 
