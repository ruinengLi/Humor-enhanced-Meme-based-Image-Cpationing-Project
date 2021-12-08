#Modified by 李睿能

#some basic imports and setups
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = os.getcwd()
image_dir = os.path.join(current_dir, 'memes')

import tensorflow as tf
import numpy as np

class AlexNet(object):

  def __init__(self, x, keep_prob, num_classes, finetune_layer,skip_layer, hidden_size,
               weights_path = 'DEFAULT'):

    # Parse input arguments into class variables
    self.X = x
    self.HIDDEN_SIZE = hidden_size
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    self.FINETUNE_LAYER = finetune_layer
    self.SKIP_LAYER = skip_layer

    if weights_path == 'DEFAULT':
      self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
    else:
      self.WEIGHTS_PATH = weights_path

    # Call the create function to build the computational graph of AlexNet
    self.create()

  def create(self):

    # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
    conv1 = conv(self.X, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
    pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
    norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')

    # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
    conv2 = conv(norm1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2')
    pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
    norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')

    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(norm2, 3, 3, 384, 1, 1, name = 'conv3')

    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4')

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool5, [-1, 6*6*256])
    self.fc6 = fc(flattened, 6*6*256, 4096, name='fc6')  #for the moment have only fc7 be trainable
    dropout6 = dropout(self.fc6, self.KEEP_PROB)

    #Now editing for MemeProject: Change last layer (fc8)  to trainable
    # 7th Layer: FC (w ReLu) -> Dropout
    fc7 = fc(dropout6, 4096, self.HIDDEN_SIZE, name = 'fc7',relu=False)
    dropout7 = dropout(fc7, self.KEEP_PROB)


    # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
    fc8 = fc(dropout7, 512, self.HIDDEN_SIZE, relu = False, name='fc8')



  def load_initial_weights(self, session):
    """
    As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come
    as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of
    dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
    need a special load function
    """

    # Load the weights into memory
    weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()

    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:

      # Check if the layer is one of the layers that should be reinitialized
      if op_name not in self.FINETUNE_LAYER and op_name not in self.SKIP_LAYER:

        with tf.variable_scope(op_name, reuse = True):

          # Loop over list of weights/biases and assign them to their corresponding tf variable
          for data in weights_dict[op_name]:

            # Biases
            if len(data.shape) == 1:

              var = tf.get_variable('biases', trainable = False)
              session.run(var.assign(data))

            # Weights
            else:

              var = tf.get_variable('weights', trainable = False)
              session.run(var.assign(data))
      elif op_name not in self.SKIP_LAYER:

        with tf.variable_scope(op_name, reuse = True):

            # Loop over list of weights/biases and assign them to their corresponding tf variable
          for data in weights_dict[op_name]:

            # Biases
            if len(data.shape) == 1:

              var = tf.get_variable('biases', trainable = True)
              session.run(var.assign(data))

            # Weights
            else:

              var = tf.get_variable('weights', trainable = True)
              session.run(var.assign(data))




"""
Predefine all necessary layers for the AlexNet
"""
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
  """
  Adapted from: https://github.com/ethereon/caffe-tensorflow
  """
  # Get number of input channels
  input_channels = int(x.get_shape()[-1])

  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k,
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)

  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters])
    biases = tf.get_variable('biases', shape = [num_filters])


    if groups == 1:
      conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
      # Split input and weights and convolve them separately
      input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
      weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
      output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]

      # Concat the convolved output together again
      conv = tf.concat(axis = 3, values = output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name)

    return relu

def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:

    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)

    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)
      return relu
    else:
      return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
  return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
                                            beta = beta, bias = bias, name = name)

def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)

#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

#create model with default config ( == no skip_layer and 1000 units in the last layer)
model = AlexNet(x, keep_prob, 1000,[],['fc7','fc8'],512,weights_path='bvlc_alexnet.npy') #maybe need to put fc8 in skip_layers

#define activation of last layer as score
score = model.fc6

os.getcwd()
with open('ordered_memes.txt','r') as f:
    img_files = f.readlines()
img_files = [os.path.join(image_dir, f) for f in img_files] # add path to each file
img_files = [img_file.replace('\n','') for img_file in img_files]
print(img_files[0],img_files[-1])
with open('Captions.txt','r') as f:
    captions = f.readlines()
#captions = list(set(captions))
captions = [s.lower() for s in captions]
deleters = []
for i,capt in enumerate(captions):
    if ' - ' not in capt or ' - -' in capt:
        deleters.append(i)
for i,delete in enumerate(deleters):
    del captions[delete-i]
data_memes = []
data_captions = []
data_meme_names = [] #just to check captions have been paired correctly
counter = 0
passed = 0


#Doing everything in one script: (the fc6 vectors are quite sparse)
with tf.Session() as sess:
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # Load the pretrained weights into the model
    model.load_initial_weights(sess)
    
    for i,meme in enumerate(img_files):
        #meme_name = meme.replace('/Users/ALP/Desktop/Stanford/CS224n/MemeProject/memes/','')
        #meme_name = meme_name.replace('.jpg','').lower()
        #meme_name = meme_name.replace('-',' ')
        img = Image.open(meme)
        try:
            img.thumbnail((227, 227), Image.ANTIALIAS)
            #img = img.resize((227,227))
            #use img.thumbnail for square images, img.resize for non square
            assert np.shape(img) == (227, 227, 3)
        except AssertionError:
            img = img.resize((227,227))
            print('sizing error')
        
        # Subtract the ImageNet mean
        img = img - imagenet_mean #should probably change this
        
        # Reshape as needed to feed into model
        img = img.reshape((1,227,227,3))

        meme_vector = sess.run(score, feed_dict={x: img, keep_prob: 1}) #[1,4096]
        meme_vector = np.reshape(meme_vector,[4096])
        assert np.shape(meme_vector) == (4096,)
        
        match = []
        meme_name = captions[counter].split(' - ')[0]
        image_name = img_files[i].replace('/output/meme/','')
        image_name = image_name.replace('.jpg','')
        image_name = image_name.replace('-',' ')
        #print(meme_name,image_name)
        try:
            assert SequenceMatcher(a=meme_name.replace(' ',''),b=image_name.replace(' ','')).ratio() >= 0.75 or image_name in meme_name or meme_name in image_name
        except AssertionError:
            passed+=1
            continue
        
        while SequenceMatcher(a=meme_name.replace(' ',''),b=captions[counter].split(' - ')[0].replace(' ','')).ratio() >= 0.75 or meme_name in captions[counter].split(' - ')[0] or captions[counter].split(' - ')[0] in meme_name: 
            if counter==len(captions)-1:
                match.append(captions[counter].split(' - ')[-1])
                break
            elif captions[counter] == captions[counter].split(' - ')[-1]:
                counter += 1
            else:
                match.append(captions[counter].split(' - ')[-1])
                counter += 1
                
        
        #now save in tfrecords format, or prepare for that action
        meme_vectors = [meme_vector for cap in match]
        image_names = [image_name for cap in match]
        assert len(meme_vectors) == len(match)
        data_memes.extend(meme_vectors)
        data_captions.extend(match)
        data_meme_names.extend(image_names)

        if i % 100 == 0:
            print(i,len(data_memes),len(data_captions),len(data_meme_names))

print(passed)
print(len(data_memes))

with open('ordered_memes.txt','r') as f:
    img_files = f.readlines()
img_files = [os.path.join(image_dir, f) for f in img_files] # add path to each file
img_files = [img_file.replace('\n','') for img_file in img_files]
print(img_files[0],img_files[-1])
with open('CaptionsClean.txt','r') as f:
    captions = f.readlines()

data_memes = []
data_captions = []
data_labels = []
counter = 0
passed = 0


for i,meme in enumerate(img_files):
    
    match = []
    meme_name = captions[counter].split(' - ')[0]
    image_name = meme.replace('/memes/','')
    image_name = image_name.replace('.jpg','')
    image_name = image_name.replace('-',' ')
    #print(meme_name,image_name)
    try:
        assert SequenceMatcher(a=meme_name.replace(' ',''),b=image_name.replace(' ','')).ratio() >= 0.75 or image_name in meme_name or meme_name in image_name
    except AssertionError:
        passed+=1
        continue
        
    while SequenceMatcher(a=meme_name.replace(' ',''),b=captions[counter].split(' - ')[0].replace(' ','')).ratio() >= 0.75 or meme_name in captions[counter].split(' - ')[0] or captions[counter].split(' - ')[0] in meme_name: 
        if counter==len(captions)-1:
            match.append(captions[counter].split(' - ')[-1])
            break
        elif captions[counter] == captions[counter].split(' - ')[-1]:
            counter += 1
        else:
            match.append(captions[counter].split(' - ')[-1])
            counter += 1
                
        
    #now save in tfrecords format, or prepare for that action
    meme_images = [meme for cap in match]
    meme_labels = [meme_name for cap in match]
    assert len(meme_images) == len(match)
    data_memes.extend(meme_images)
    data_captions.extend(match)
    data_labels.extend(meme_labels)

    if i % 100 == 0:
        print(i,len(data_memes),len(data_captions),len(meme_labels))
print(passed)
print(len(data_memes))

counter = 0
R = []
idxs = []

for i,capt in enumerate(data_captions):
    if i % 190 == 0: 
        start = ' '.join(capt.split()[0:2])
        counter = 0
    if SequenceMatcher(a=start,b=' '.join(capt.split()[0:2])).ratio() >= 0.9:
        counter += 1 
    if counter >20 and start not in R:
        R.append(start)
        idxs.append(i)

c = list(zip(data_memes, data_captions, data_labels))
no_repeats = []
# order preserving
def idfun(x): return x

seen = {}
no_repeats = []
for item in c:
    marker = idfun(item[1])
    # in old Python versions:
    # if seen.has_key(marker)
    # but in new ones:
    if marker in seen: continue
    seen[marker] = 1
    no_repeats.append(item)

from random import shuffle
shuffle(no_repeats)
memes_shuffled, captions_shuffled, labels_shuffled = zip(*no_repeats)
memes_shuffled = list(memes_shuffled)
captions_shuffled = list(captions_shuffled)
labels_shuffled = list(labels_shuffled)

import re
word_captions = []
for capt in captions_shuffled + labels_shuffled: #include labels_shuffled here for glove averages
    words = re.findall(r"[\w']+|[.,:!?;'><(){}%$#£@-_+=|\/~`^&*]", capt)
    word_captions.append(words)

from collections import Counter
print("Creating vocabulary.")
counter = Counter()
for c in word_captions:
    counter.update(c)
print("Total words:", len(counter))

# Filter uncommon words and sort by descending count.
word_counts = [x for x in counter.items() if x[1] >= 3]
word_counts.sort(key=lambda x: x[1], reverse=True)
print("Words in vocabulary:", len(word_counts))

# Create the vocabulary dictionary.
reverse_vocab = [x[0] for x in word_counts]
#unk_id = len(reverse_vocab)
vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

import zipfile
EMBEDDING_DIMENSION=300 # Available dimensions for 6B data is 50, 100, 200, 300
data_directory = current_dir

PAD_TOKEN = 0

word2idx = { 'PAD': PAD_TOKEN } # dict so we can lookup indices for tokenising our text later from string to sequence of integers
weights = []
index_counter = 0

with open('glove.6B.300d.txt','r') as file:
    for index, line in enumerate(file):
        values = line.split() # Word and weights separated by space
        word = values[0] # Word is first symbol on each line
        if word in vocab_dict:
            index_counter += 1
            word_weights = np.asarray(values[1:], dtype=np.float32) # Remainder of line is weights for word
            word2idx[word] = index_counter # PAD is our zeroth index so shift by one
            weights.append(word_weights)
        if index % 100000 == 0:
            print(index)
        if index + 1 == 2000000:
            break

EMBEDDING_DIMENSION = len(weights[0])
# Insert the PAD weights at index 0 now we know the embedding dimension
weights.insert(0, np.zeros(EMBEDDING_DIMENSION))

# Append unknown and pad to end of vocab and initialize as random #maybe include start and end token here
UNKNOWN_TOKEN=len(weights)
#include * token as its very common? why not
word2idx['*'] = UNKNOWN_TOKEN
word2idx['UNK'] = UNKNOWN_TOKEN + 1
word2idx['<S>'] = UNKNOWN_TOKEN + 2
word2idx['</S>'] = UNKNOWN_TOKEN + 3
weights.append(np.random.randn(EMBEDDING_DIMENSION)*0.5)
weights.append(np.random.randn(EMBEDDING_DIMENSION)*0.5)
weights.append(np.random.randn(EMBEDDING_DIMENSION)*0.5)
weights.append(np.random.randn(EMBEDDING_DIMENSION)*0.5)

# Construct our final vocab
weights = np.asarray(weights, dtype=np.float32)

VOCAB_SIZE=weights.shape[0]
print(VOCAB_SIZE)

#Save Vocabulary IF NEW
with tf.gfile.FastGFile('vocab_averages.txt', "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word2idx.items()]))
print("Wrote vocabulary file:", 'vocab_averages.txt')

np.savetxt('embedding_matrix',weights)

eval_labels = []
eval_memes = []

for idx in eval_examples:
    eval_memes.append(data_memes[idx])
    eval_labels.append(data_labels[idx])


import re
token_captions = []
for capt in captions_shuffled:
    token_caption = []
    token_caption.append(word2idx['<S>'])
    words = re.findall(r"[\w']+|[.,:!?;'><(){}%$#£@-_+=|\/~`^&*]", capt)
    for word in words:
        try:
            token = word2idx[word]
        except KeyError:
            token = word2idx['UNK']
        token_caption.append(token)
    token_caption.append(word2idx['</S>'])
    token_captions.append(token_caption)
    
#for training labels
token_labels = []
for capt in labels_shuffled:
    token_caption = []
    words = re.findall(r"[\w']+|[.,:!?;'><(){}%$#£@-_+=|\/~`^&*]", capt)
    for word in words:
        try:
            token = word2idx[word]
        except KeyError:
            token = word2idx['UNK']
        token_caption.append(token)
    token_labels.append(token_caption)
    
#for eval set
eval_tokens = []
for capt in eval_captions:
    token_caption = []
    token_caption.append(word2idx['<S>'])
    words = re.findall(r"[\w']+|[.,:!?;'><(){}%$#£@-_+=|\/~`^&*]", capt)
    for word in words:
        try:
            token = word2idx[word]
        except KeyError:
            token = word2idx['UNK']
        token_caption.append(token)
    #token_caption.append(word2idx['</S>'])
    eval_tokens.append(token_caption)
    
#for eval labels
token_labels_eval = []
for capt in eval_labels:
    token_caption = []
    words = re.findall(r"[\w']+|[.,:!?;'><(){}%$#£@-_+=|\/~`^&*]", capt)
    for word in words:
        try:
            token = word2idx[word]
        except KeyError:
            token = word2idx['UNK']
        token_caption.append(token)
    token_labels_eval.append(token_caption)

memes_shuffled = list(memes_shuffled)
captions_shuffled = list(captions_shuffled)

deleters = []
for i,ting in enumerate(token_captions):
    if len(ting) == 2:
        deleters.append(i)

for i,ting in enumerate(deleters):
    print(ting)
    del captions_shuffled[ting-i]
    del memes_shuffled[ting-i]
    del token_captions[ting-i]
    del labels_shuffled[ting-i]
    del token_labels[ting-i]
deleters = []
for i,ting in enumerate(token_captions):
    if len(ting) == 2:
        deleters.append(i)

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

memes_shuffled_int = []
for i,meme in enumerate(memes_shuffled):
    memes_shuffled_int.append(np.fromstring(meme*100000,dtype=int))
print(memes_shuffled_int[10][:100])

eval_memes_int = []
for i,meme in enumerate(eval_memes):
    eval_memes_int.append(np.fromstring(meme*100000,dtype=int))

import sys
train_filename = 'train.tfrecordsALEX'  # address to save the TFRecords file
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(memes_shuffled_int)):
    if not i % 100:
        print('Train data: {}/{}'.format(i, len(memes_shuffled_int)))
        sys.stdout.flush()
    context = tf.train.Features(feature={
          "train/meme": _bytes_feature(memes_shuffled_int[i].tostring()), 
      })
    feature_lists = tf.train.FeatureLists(feature_list={
          "train/captions": _int64_feature_list(token_captions[i])
      })
    sequence_example = tf.train.SequenceExample(
          context=context, feature_lists=feature_lists)
    
    writer.write(sequence_example.SerializeToString())
    
writer.close()
sys.stdout.flush()



