pip install Pillow

# Clear the file
> train_unet.py

# Populate the file
echo "from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
import tensorflow as tf
TRAIN_ITER = 5
TRAIN_EPOCH = 1
BS = 16
if __name__ == '__main__':
    nx = 572
    ny = 572
    generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20)
    net = unet.Unet(channels=generator.channels, n_class=generator.n_class, layers=3, features_root=16)
    trainer = unet.Trainer(net, batch_size=BS, optimizer='momentum', opt_kwargs=dict(momentum=0.2))
    path = trainer.train(generator, './unet_trained', training_iters=TRAIN_ITER, epochs=TRAIN_EPOCH, display_step=1)
    print('Done training unet')" > train_unet.py

git add train_unet.py
git commit -m "Adding train script"
