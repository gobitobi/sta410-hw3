import tensorflow as tf


d = 3
@tf.function(input_signature=(tf.TensorSpec(shape=[d], dtype=tf.float32), ))
def schwefel(x):
    y = tf.math.reduce_sum(x*tf.math.sin(tf.math.sqrt(tf.math.abs(x))))
    return 418.9829*x.shape[0] - y

def newtons_method(f, x0, K=10, eps=1e-7):
    
    '''
    Newton's Method with TensorFlow
    
    f   : @tf.function(input_signature=(tf.TensorSpec(shape=[d], dtype=tf.float32), ))
    x0  : [x0_0, x0_1, ..., x0_(d-1) initialization 
    K   : (default 10) number of Newton Method steps
    eps : (default 0.01) stopping criterion `||x_k - x_(k-1)||_2<eps`
    
    returns x_k.numpy().tolist()+[f(x_k).numpy()]
            where `_K` indicates the stopping criteria has been met
    '''

    x_k = tf.Variable(x0)
    
    # <complete>
    
    return x_k.numpy(),f(x_k).numpy()  