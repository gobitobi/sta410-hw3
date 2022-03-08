import tensorflow as tf

# https://www.tensorflow.org/guide/function#basics
tf_Variable = tf.TensorSpec(shape=[], dtype=tf.float32)
@tf.function(input_signature=(tf_Variable, tf_Variable, ))
def eggholder(x1,x2):
    y = -(x2+47)*tf.math.sin(tf.sqrt(tf.math.abs(x2+x1/2+47)))
    return y - x1*tf.math.sin(tf.sqrt(tf.math.abs(x1-(x2+47))))

def nonlinear_gauss_seidel(f, x0, x_constraint, N=20, a=0.1, eps=0.1, K=100):
    
    '''
    Nonlinear Gauss-Seidel using Univariate Gradient Descent with TensorFlow
    
    f   : @tf.function(input_signature=(tf_Variable, tf_Variable, ))
    x0  : (float,float) initialization 
    x_constraint : [[min_x1,max_x1],[min_x2,max_x2]) 
                   xi_t exceeding bounds is reassinged exceeded bound endpoint                   
    N   : (default 100) number of Gauss-Seidel cycles
    a   : (default 0.1) gradient descent step size factor
    eps : (default 0.1) stopping criterion `|tape.gradient(y, x2)|<eps`
    K   : (default 100) stopping criterion maximum number of gradient descent steps
    
    returns x1_N.numpy(),x2_N.numpy(),f(x1_N,x2_N).numpy()
            where `_N` indicates completion of Nonlinear Gauss-Seidel cycles
    '''
    
    x1 = tf.Variable(x0[0])
    x2 = tf.Variable(x0[1])
    
    # <complete>
                    
    return x1.numpy(),x2.numpy(),f(x1,x2).numpy()