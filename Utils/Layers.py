import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np

paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.1

def getParamId():
	global paramId
	paramId += 1
	return paramId

def setIta(ITA):
	ita = ITA

def setBiasDefault(val):
	global biasDefault
	biasDefault = val

def getParam(name):
	return params[name]

def addReg(name, param):
	global regParams
	if name not in regParams:
		regParams[name] = param
	else:
		print('ERROR: Parameter already exists')

def addParam(name, param):
	global params
	if name not in params:
		params[name] = param

def defineRandomNameParam(shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	name = 'defaultParamName%d'%getParamId()
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def defineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	global params
	global regParams
	assert name not in params, 'name %s already exists' % name
	if initializer == 'xavier':
		ret = tf.get_variable(name=name, dtype=dtype, shape=shape,
			initializer=xavier_initializer(dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'trunc_normal':
		ret = tf.get_variable(name=name, initializer=tf.random.truncated_normal(shape=[int(shape[0]), shape[1]], mean=0.0, stddev=0.03, dtype=dtype))
	elif initializer == 'zeros':
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=tf.zeros(shape=shape, dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'ones':
		ret = tf.get_variable(name=name, dtype=dtype, initializer=tf.ones(shape=shape, dtype=tf.float32), trainable=trainable)
	elif not isinstance(initializer, str):
		ret = tf.get_variable(name=name, dtype=dtype,
			initializer=initializer, trainable=trainable)
	else:
		print('ERROR: Unrecognized initializer')
		exit()
	params[name] = ret
	if reg:
		regParams[name] = ret
	return ret

def getOrDefineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True, reuse=False):
	global params
	global regParams
	if name in params:
		assert reuse, 'Reusing Param %s Not Specified' % name
		if reg and name not in regParams:
			regParams[name] = params[name]
		return params[name]
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def BN(inp, name=None):
	global ita
	dim = inp.get_shape()[1]
	name = 'defaultParamName%d'%getParamId()
	scale = tf.Variable(tf.ones([dim]))
	shift = tf.Variable(tf.zeros([dim]))
	# 计算均值和方差
	fcMean, fcVar = tf.nn.moments(inp, axes=[0])
	# 滑动平均模型，他使用指数衰减来计算变量的移动平均值
	ema = tf.train.ExponentialMovingAverage(decay=0.5)
	emaApplyOp = ema.apply([fcMean, fcVar])
	with tf.control_dependencies([emaApplyOp]):
		mean = tf.identity(fcMean) # 返回一个和input一样的新的tensor
		var = tf.identity(fcVar)
	ret = tf.nn.batch_normalization(inp, mean, var, shift,
		scale, 1e-8)
	return ret

def FC(inp, outDim, name=None, useBias=False, activation=None, reg=False, useBN=False, dropout=None, initializer='xavier', reuse=False, biasReg=False, biasInitializer='zeros'):
	global params
	global regParams
	global leaky
	inDim = inp.get_shape()[1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	W = getOrDefineParam(temName, [inDim, outDim], reg=reg, initializer=initializer, reuse=reuse)
	if dropout != None:
		ret = tf.nn.dropout(inp, rate=dropout) @ W
	else:
		ret = inp @ W
	if useBias:
		ret = Bias(ret, name=name, reuse=reuse, reg=biasReg, initializer=biasInitializer)
	if useBN:
		ret = BN(ret)
	if activation != None:
		ret = Activate(ret, activation)
	return ret

def Bias(data, name=None, reg=False, reuse=False, initializer='zeros'):
	inDim = data.get_shape()[-1]
	temName = name if name!=None else 'defaultParamName%d'%getParamId()
	temBiasName = temName + 'Bias'
	bias = getOrDefineParam(temBiasName, inDim, reg=False, initializer=initializer, reuse=reuse)
	if reg:
		regParams[temBiasName] = bias
	return data + bias

def ActivateHelp(data, method):
	if method == 'relu':
		ret = tf.nn.relu(data)
	elif method == 'sigmoid':
		ret = tf.nn.sigmoid(data)
	elif method == 'tanh':
		ret = tf.nn.tanh(data)
	elif method == 'softmax':
		ret = tf.nn.softmax(data, axis=-1)
	elif method == 'leakyRelu':
		ret = tf.maximum(leaky*data, data)
	elif method == 'twoWayLeakyRelu6':
		temMask = tf.to_float(tf.greater(data, 6.0))
		ret = temMask * (6 + leaky * (data - 6)) + (1 - temMask) * tf.maximum(leaky * data, data)
	elif method == '-1relu':
		ret = tf.maximum(-1.0, data)
	elif method == 'relu6':
		ret = tf.maximum(0.0, tf.minimum(6.0, data))
	elif method == 'relu3':
		ret = tf.maximum(0.0, tf.minimum(3.0, data))
	else:
		raise Exception('Error Activation Function')
	return ret

def Activate(data, method, useBN=False):
	global leaky
	if useBN:
		ret = BN(data)
	else:
		ret = data
	ret = ActivateHelp(ret, method)
	return ret

def Regularize(names=None, method='L2'):
	ret = 0
	if method == 'L1':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.abs(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.abs(regParams[name]))
	elif method == 'L2':
		if names != None:
			for name in names:
				ret += tf.reduce_sum(tf.square(getParam(name)))
		else:
			for name in regParams:
				ret += tf.reduce_sum(tf.square(regParams[name]))
	return ret

def Dropout(data, rate):
	if rate == None:
		return data
	else:
		return tf.nn.dropout(data, rate=rate)

class ScaledDotProductAttention(object):
    def __init__(self, d_k):
        self.d_k = d_k
    
    def attention(self, Q, K, V, attn_mask=None):
        with tf.name_scope('scaled_attention'): 
            # batch_size,head_num, candidate_num, candidate_num
            scores = tf.matmul(Q, tf.transpose(K,perm=[0,1,3,2])) / np.sqrt(self.d_k)
            scores = tf.exp(scores)
            if attn_mask is not None:
                scores = scores * attn_mask
            # batch_size,head_num, candidate_num, 1
            attn = scores / (tf.expand_dims(tf.reduce_sum(scores, axis=-1),-1) + 1e-8) # 归一化
            context = tf.matmul(attn, V)
            return context, attn

class MultiHeadSelfAttention(object):
    def __init__(self, d_model, num_attention_heads):
        self.d_model = d_model # embedding_size
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads #16
        self.d_v = d_model // num_attention_heads
        
    def attention(self, Q, K=None, V=None, length=None):
        """
        Q:batch_size,candidate_num,embedding_size
        return : batch_size,candidate_num,embedding_size
        """
        with tf.name_scope('multihead_selfattention'): 
            if K is None:
                K = Q
            if V is None:
                V = Q
            batch_size = Q.shape[0]
            W_Q = tf.layers.dense(Q, self.d_model,kernel_initializer=tf.contrib.layers.xavier_initializer( uniform=True, seed=None, dtype=tf.float32 ))
            # batch_size, candidate_num, num_attention_heads,d_k  ;;divide into groups whose num is num_attention_heads
            # batch_size, num_attention_heads, candidate_num,d_k
            q_s = tf.transpose(tf.reshape(W_Q,[batch_size, -1, self.num_attention_heads,self.d_k]),perm=[0,2,1,3])
            W_K = tf.layers.dense(K, self.d_model,kernel_initializer=tf.contrib.layers.xavier_initializer( uniform=True, seed=None, dtype=tf.float32 ))
            k_s = tf.transpose(tf.reshape(W_K,[batch_size, -1, self.num_attention_heads,self.d_k]),perm=[0,2,1,3])
            W_V = tf.layers.dense(V, self.d_model,kernel_initializer=tf.contrib.layers.xavier_initializer( uniform=True, seed=None, dtype=tf.float32 ))
            v_s = tf.transpose(tf.reshape(W_V,[batch_size, -1, self.num_attention_heads,self.d_v]),perm=[0,2,1,3])
            # batch_size,head_num, candidate_num, d_k
            context, attn = ScaledDotProductAttention(self.d_k).attention(q_s, k_s, v_s)#,attn_mask)
            # batch_size,candidate_num,embedding_size
            context= tf.reshape(tf.transpose(context,perm=[0,2,1,3]),[batch_size, -1, self.num_attention_heads*self.d_v])
            return context

