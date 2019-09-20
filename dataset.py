import tensorflow as tf
import tensorflow.data as tfd
import tensorflow.image as tfi
import tensorflow.random as tfr

import itertools as it
import operator as op

import diux.data_utilities.wv_util as wvu
import core.preprocessor as pp


def main():
    D = fImPatches()
    iD = D.make_one_shot_iterator()
    d = next(iD)
    print(d.shape)

def fImPatches(
    img_dir='/home/jl/data/xview/train_images/tif/*.tif'
    ,labelfile = "/home/jl/data/xview/xView_train.geojson"
    ,stride:int=500
    ,patch_dim:int=1000
    ,zoom:float=0.25
    ,out_dim:int=300
    )->tfd.Dataset:
    def fPatches(img:tf.Tensor)-> tf.Tensor:
        def fPatchAug(patch:tf.Tensor)->tf.Tensor:
            patch = tf.cast(tfi.resize(patch, tf.cast(tf.cast(patch.shape[:2], tf.float32) * tfr.uniform([], 1 - zoom, 1 + zoom), tf.int32)),tf.uint8)
            patch = tfi.random_crop(patch, (out_dim, out_dim,patch.shape[-1]))
            patch = tfi.random_flip_left_right(patch)
            patch = tfi.random_flip_up_down(patch)
            patch = tfi.rot90(patch, tfr.uniform([], 0, 3, tf.int32))
            return patch
        img = tfi.random_crop(img,tf.concat([stride*(tf.cast(tf.shape(img)[:2],tf.int32) // stride),tf.shape(img)[2:]],0))
        patches = tf.reshape(tfi.extract_image_patches(tf.expand_dims(img,0),(1,patch_dim,patch_dim,1),(1,stride,stride,1),(1,1,1,1),"VALID"),(-1,patch_dim,patch_dim,3))
        patches = tf.map_fn(fPatchAug,patches)
        return patches
    #coords, chips, classes = wvu.get_labels(labelfile)
    #CCC = it.groupby(sorted(zip(chips, coords, classes)), op.itemgetter(0))

    D = tfd.Dataset.list_files(img_dir,shuffle=True).repeat()
    D = D.map(tf.io.read_file).map(tf.io.decode_png)
    D = D.filter(lambda x:tf.reduce_mean(tf.cast(tf.equal(x,0),tf.float32))<.9)
    D = D.map(fPatches)
    D = D.flat_map(lambda x: tfd.Dataset.from_tensor_slices(x)).shuffle(100)
    return D




if __name__ == "__main__": main()
