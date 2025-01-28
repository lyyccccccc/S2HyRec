from ast import arg
from curses import meta
from site import USER_BASE
from matplotlib.cbook import silent_list
import matplotlib.pyplot as plt
from Params import args
import Utils.Layers as NNs
from  Utils.Layers import FC, Regularize, Activate,MultiHeadSelfAttention
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
import Utils.Logger as logger
import numpy as np
from Utils.Logger import log
from DataHandler import negSamp,negSamp_fre, transpose, DataHandler, transToLsts
from random import randint

class S2HyRec:
    def __init__(self, sess, handler):
        self.sess = sess
        self.handler = handler

        print('USER', args.user, 'ITEM', args.item)
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'HR', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret
    
    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            init = tf.global_variables_initializer()
            self.sess.run(init)
            log('Variables Inited')
        
        maxndcg=0.0
        maxres=dict()
        maxepoch=0
        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, test))
            if test:
                reses = self.testEpoch()
                log(self.makePrint('Test', ep, reses, test))
            if ep % args.tstEpoch == 0 and reses['NDCG10']>maxndcg:
                self.saveHistory()
                maxndcg=reses['NDCG10']
                maxres=reses
                maxepoch=ep
            
            print()

        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        log(self.makePrint('max', maxepoch, maxres, True))
        
    def makeTimeEmbed(self):
        divTerm = 1 / (10000 ** (tf.range(0, args.latdim * 2, 2, dtype=tf.float32) / args.latdim))
        pos = tf.expand_dims(tf.range(0, self.maxTime, dtype=tf.float32), axis=-1)
        sine = tf.expand_dims(tf.math.sin(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
        cosine = tf.expand_dims(tf.math.cos(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
        timeEmbed = tf.reshape(tf.concat([sine, cosine], axis=-1), [self.maxTime, args.latdim*2]) / 4.0
        return timeEmbed
    def messagePropagate(self, srclats, mat, type='user'):
        srcNodes = tf.squeeze(tf.slice(mat.indices, [0, 1], [-1, 1])) 
        tgtNodes = tf.squeeze(tf.slice(mat.indices, [0, 0], [-1, 1])) 
        srcEmbeds = tf.nn.embedding_lookup(srclats, srcNodes) 
        lat=tf.pad(tf.math.segment_sum(srcEmbeds, tgtNodes),[[0,100],[0,0]]) 
        if(type=='user'):
            lat=tf.nn.embedding_lookup(lat,self.users)
        else:
            lat=tf.nn.embedding_lookup(lat,self.items)
        return Activate(lat, self.actFunc)
    def edgeDropout(self, mat):
        def dropOneMat(mat):
            indices = mat.indices
            values = mat.values
            shape = mat.dense_shape
            newVals = tf.nn.dropout(tf.cast(values,dtype=tf.float32), self.keepRate)
            return tf.sparse.SparseTensor(indices, tf.cast(newVals,dtype=tf.int32), shape)
        return dropOneMat(mat)
    def hyperPropagate(self, lats, adj):
        lat1 = Activate(tf.transpose(adj) @ lats, self.actFunc)
        lat2 = tf.transpose(FC(tf.transpose(lat1), args.hyperNum, activation=self.actFunc)) + lat1
        lat3 = tf.transpose(FC(tf.transpose(lat2), args.hyperNum, activation=self.actFunc)) + lat2
        lat4 = tf.transpose(FC(tf.transpose(lat3), args.hyperNum, activation=self.actFunc)) + lat3
        ret = Activate(adj @ lat4, self.actFunc)
        return ret
    def sslLoss(self,proj_final_temporal, proj_global):
            proj_global = tf.nn.l2_normalize(proj_global, axis=1)
            proj_final_temporal = tf.nn.l2_normalize(proj_final_temporal, axis=1)
            
            posScore = tf.exp(tf.reduce_sum(proj_global * proj_final_temporal, axis=1) / args.temp)
            negScore = tf.reduce_sum(tf.exp(proj_final_temporal @ tf.transpose(proj_global) / args.temp), axis=1)
            sslLoss = tf.reduce_sum(-tf.log(posScore / (negScore + 1e-8) + 1e-8))
            return sslLoss
    
    def S2Hy(self):
        temporal_user,temporal_item=list(),list()
        uEmbed=NNs.defineParam('uEmbed', [args.graphNum, args.user, args.latdim], reg=True)
        iEmbed=NNs.defineParam('iEmbed', [args.graphNum, args.item, args.latdim], reg=True)
        uhyper = NNs.defineParam('uhyper', [args.latdim, args.hyperNum], reg=True)
        ihyper = NNs.defineParam('ihyper', [args.latdim, args.hyperNum], reg=True)	
        posEmbed=NNs.defineParam('posEmbed', [args.pos_length, args.latdim], reg=True)
        pos= tf.tile(tf.expand_dims(tf.range(args.pos_length),axis=0),[args.batch,1])
        self.items=tf.range(args.item)
        self.users=tf.range(args.user)
        self.timeEmbed=NNs.defineParam('timeEmbed', [self.maxTime+1, args.latdim], reg=True)
        overall_u_embed = tf.reduce_mean(uEmbed, axis=0) 
        overall_i_embed = tf.reduce_mean(iEmbed, axis=0)
        uuHyper = (overall_u_embed @ uhyper)
        iiHyper = (overall_i_embed @ ihyper)
        uuHyper_dropped = tf.nn.dropout(uuHyper, self.keepRate)
        iiHyper_dropped = tf.nn.dropout(iiHyper, self.keepRate)
        global_user = self.hyperPropagate(overall_u_embed, uuHyper_dropped)
        global_item = self.hyperPropagate(overall_i_embed, iiHyper_dropped)

        for k in range(args.graphNum):
            embs0=[uEmbed[k]]
            embs1=[iEmbed[k]]
            prev_gcn_u = embs0[-1]
            prev_gcn_i = embs1[-1]
            slice_gnnULats = []
            slice_gnnILats = []
            for i in range(args.gnn_layer):
                a_emb0= self.messagePropagate(prev_gcn_i,self.edgeDropout(self.subAdj[k]),'user')
                a_emb1= self.messagePropagate(prev_gcn_u,self.edgeDropout(self.subTpAdj[k]),'item')
                slice_gnnULats.append(a_emb0)
                slice_gnnILats.append(a_emb1)
                prev_gcn_u = a_emb0
                prev_gcn_i = a_emb1
            gcn_slice_u = tf.add_n(slice_gnnULats) / len(slice_gnnULats)
            gcn_slice_i = tf.add_n(slice_gnnILats) / len(slice_gnnILats)
            temporal_user.append(gcn_slice_u)
            temporal_item.append(gcn_slice_i)
        temporal_user = tf.stack(temporal_user, axis=0)  
        temporal_item = tf.stack(temporal_item, axis=0) 
        temporal_user_sequence = tf.transpose(temporal_user, perm=[1, 0, 2])
        temporal_item_sequence = tf.transpose(temporal_item, perm=[1, 0, 2])
        self.multihead_self_attention = MultiHeadSelfAttention(args.latdim,args.num_attention_heads)
        temporal_user_sequence_all = self.multihead_self_attention.attention(tf.contrib.layers.layer_norm(temporal_user_sequence))
        temporal_item_sequence_all = self.multihead_self_attention.attention(tf.contrib.layers.layer_norm(temporal_item_sequence))
        final_temporal_user = tf.reduce_mean(temporal_user_sequence_all,axis=1)
        final_temporal_item = tf.reduce_mean(temporal_item_sequence_all,axis=1)
        final_user= (1-args.alpha)*final_temporal_user + args.alpha*global_user
        final_item= (1-args.alpha)*final_temporal_item + args.alpha*global_item
        iEmbed_att=final_item

        self.multihead_self_attention_sequence = list()
        for i in range(args.att_layer):
            self.multihead_self_attention_sequence.append(MultiHeadSelfAttention(args.latdim,args.num_attention_heads))
        sequence_dependency=tf.contrib.layers.layer_norm(tf.matmul(tf.expand_dims(self.mask,axis=1),tf.nn.embedding_lookup(iEmbed_att,self.sequence)))
        sequence_dependency+=tf.contrib.layers.layer_norm(tf.matmul(tf.expand_dims(self.mask,axis=1),tf.nn.embedding_lookup(posEmbed,pos)))
        for i in range(args.att_layer):
            sequence_dependency1=self.multihead_self_attention_sequence[i].attention(tf.contrib.layers.layer_norm(sequence_dependency))
            sequence_dependency=Activate(sequence_dependency1,"leakyRelu")+sequence_dependency
        sequence_dependency_user=tf.reduce_sum(sequence_dependency,axis=1)
        pckIlat_att = tf.nn.embedding_lookup(iEmbed_att, self.iids)		
        pckUlat = tf.nn.embedding_lookup(final_user, self.uids)
        pckIlat = tf.nn.embedding_lookup(final_item, self.iids)
        preds = tf.reduce_sum(pckUlat * pckIlat, axis=-1)
        preds += tf.reduce_sum(Activate(tf.nn.embedding_lookup(sequence_dependency_user,self.uLocs_seq),"leakyRelu")* pckIlat_att,axis=-1)
        
        uniqUids, _ = tf.unique(self.uids)
        uniqIids, _ = tf.unique(self.iids)
        sslloss = 0
        W = NNs.defineRandomNameParam([args.latdim, args.latdim])
        final_temporal_user = tf.nn.embedding_lookup(final_temporal_user, uniqUids)
        global_user = tf.nn.embedding_lookup(global_user, uniqUids)
        final_temporal_item = tf.nn.embedding_lookup(final_temporal_item, uniqIids)
        global_item = tf.nn.embedding_lookup(global_item, uniqIids)
        proj_final_temporal_user = tf.nn.l2_normalize(final_temporal_user @ W, axis=1)
        proj_global_user = tf.nn.l2_normalize(global_user, axis=1)
        proj_final_temporal_item = tf.nn.l2_normalize(final_temporal_item @ W, axis=1)
        proj_global_item = tf.nn.l2_normalize(global_item, axis=1)
        uLoss = self.sslLoss(proj_final_temporal_user, proj_global_user)
        iLoss = self.sslLoss(proj_final_temporal_item, proj_global_item)
        sslloss += uLoss + iLoss
        
        return preds,sslloss

    def prepareModel(self):
        self.keepRate = tf.placeholder(dtype=tf.float32, shape=[])
        self.is_train = tf.placeholder_with_default(True, (), 'is_train')
        NNs.leaky = args.leaky
        self.actFunc = 'leakyRelu'
        adj = self.handler.trnMat
        idx, data, shape = transToLsts(adj, norm=True)
        self.adj = tf.sparse.SparseTensor(idx, data, shape)
        self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
        self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])
        self.sequence = tf.placeholder(name='sequence', dtype=tf.int32, shape=[args.batch,args.pos_length])
        self.mask = tf.placeholder(name='mask', dtype=tf.float32, shape=[args.batch,args.pos_length])
        self.uLocs_seq = tf.placeholder(name='uLocs_seq', dtype=tf.int32, shape=[None])
        self.subAdj=list()
        self.subTpAdj=list()
        for i in range(args.graphNum):
            seqadj = self.handler.subMat[i]
            idx, data, shape = transToLsts(seqadj, norm=True)
            print("1",shape)
            self.subAdj.append(tf.sparse.SparseTensor(idx, data, shape))
            idx, data, shape = transToLsts(transpose(seqadj), norm=True)
            self.subTpAdj.append(tf.sparse.SparseTensor(idx, data, shape))
            print("2",shape)
        self.maxTime=self.handler.maxTime
        self.preds, self.sslloss =self.S2Hy()
        sampNum = tf.shape(self.uids)[0] // 2
        self.posPred = tf.slice(self.preds, [0], [sampNum])# begin at 0, size = sampleNum
        self.negPred = tf.slice(self.preds, [sampNum], [-1])# 
        self.preLoss = tf.reduce_mean(tf.maximum(0.0, 1.0 - (self.posPred - self.negPred)))# +tf.reduce_mean(tf.maximum(0.0,self.negPred))
        self.regLoss = args.reg * Regularize()  + args.ssl_weight * self.sslloss
        self.loss = self.preLoss + self.regLoss
        globalStep = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

    def sampleTrainBatch(self, batIds, labelMat, timeMat, train_sample_num):
        temTst = self.handler.tstInt[batIds]
        temLabel=labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * train_sample_num
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        uLocs_seq = [None]* temlen
        sequence = [None] * args.batch
        mask = [None]*args.batch
        cur = 0				
        # utime = [[list(),list()] for x in range(args.graphNum)]
        for i in range(batch):
            posset=self.handler.sequence[batIds[i]][:-1]
            # posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
            sampNum = min(train_sample_num, len(posset))
            choose=1
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = []
                # choose = 1
                choose = randint(1,max(min(args.pred_num+1,len(posset)-3),1))
                poslocs.extend([posset[-choose]]*sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item, [self.handler.sequence[batIds[i]][-1],temTst[i]], self.handler.item_with_pop)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
                uLocs_seq[cur] = uLocs_seq[cur+temlen//2] = i
                iLocs[cur] = posloc
                iLocs[cur+temlen//2] = negloc
                cur += 1
            sequence[i]=np.zeros(args.pos_length,dtype=int)
            mask[i]=np.zeros(args.pos_length)
            posset=posset[:-choose]
            if(len(posset)<=args.pos_length):
                sequence[i][-len(posset):]=posset
                mask[i][-len(posset):]=1
            else:
                sequence[i]=posset[-args.pos_length:]
                mask[i]+=1
        uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
        uLocs_seq = uLocs_seq[:cur] + uLocs_seq[temlen//2: temlen//2 + cur]
        if(batch<args.batch):
            for i in range(batch,args.batch):
                sequence[i]=np.zeros(args.pos_length,dtype=int)
                mask[i]=np.zeros(args.pos_length)
        return uLocs, iLocs, sequence,mask, uLocs_seq# ,utime

    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss = [0] * 2
        num = len(sfIds)
        sample_num_list=[40]		
        steps = int(np.ceil(num / args.batch))
        # preLossList = [] 
        # regLossList = []  
        for s in range(len(sample_num_list)):
            for i in range(steps):
                st = i * args.batch
                ed = min((i+1) * args.batch, num)
                batIds = sfIds[st: ed]
                target = [self.optimizer, self.preLoss, self.regLoss, self.loss, self.posPred, self.negPred]
                feed_dict = {}
                uLocs, iLocs, sequence, mask, uLocs_seq= self.sampleTrainBatch(batIds, self.handler.trnMat, self.handler.timeMat, sample_num_list[s])
                
                feed_dict[self.uids] = uLocs
                feed_dict[self.iids] = iLocs
                feed_dict[self.sequence] = sequence
                feed_dict[self.mask] = mask
                feed_dict[self.is_train] = True
                feed_dict[self.uLocs_seq] = uLocs_seq
                feed_dict[self.keepRate] = args.keepRate

                res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
                preLoss, regLoss, loss, pos, neg = res[1:]
                epochLoss += loss
                epochPreLoss += preLoss
                log('Step %d/%d: preloss = %.2f, REGLoss = %.2f         ' % (i+s*steps, steps*len(sample_num_list), preLoss, regLoss), save=False, oneline=True)
                # preLossList.append(epochPreLoss / steps)
                # regLossList.append((epochLoss - epochPreLoss) / steps)
        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        return ret
        

    def sampleTestBatch(self, batIds, labelMat):
        batch = len(batIds)
        temTst = self.handler.tstInt[batIds]
        temlen = batch * args.testSize
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        uLocs_seq = [None] * temlen
        tstLocs = [None] * batch
        sequence = [None] * args.batch
        mask = [None]*args.batch
        cur = 0
        val_list=[None]*args.batch
        for i in range(batch):
            if(args.test==True):
                posloc = temTst[i]
            else:
                posloc = self.handler.sequence[batIds[i]][-1]
                val_list[i]=posloc
            rdnNegSet = np.array(self.handler.test_dict[batIds[i]+1][:args.testSize-1])-1
            locset = np.concatenate((rdnNegSet, np.array([posloc])))
            tstLocs[i] = locset
            for j in range(len(locset)):
                uLocs[cur] = batIds[i]
                iLocs[cur] = locset[j]
                uLocs_seq[cur] = i
                cur += 1
            sequence[i]=np.zeros(args.pos_length,dtype=int)
            mask[i]=np.zeros(args.pos_length)
            if(args.test==True):
                posset=self.handler.sequence[batIds[i]]
            else:
                posset=self.handler.sequence[batIds[i]][:-1]
            if(len(posset)<=args.pos_length):
                sequence[i][-len(posset):]=posset
                mask[i][-len(posset):]=1
            else:
                sequence[i]=posset[-args.pos_length:]
                mask[i]+=1
        if(batch<args.batch):
            for i in range(batch,args.batch):
                sequence[i]=np.zeros(args.pos_length,dtype=int)
                mask[i]=np.zeros(args.pos_length)
        return uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list

    def testEpoch(self):
        # epochHit, epochNdcg = [0] * 2
        epochHit5, epochNdcg5 = [0] * 2
        epochHit10, epochNdcg10 = [0] * 2
        epochHit15, epochNdcg15 = [0] * 2
        epochHit20, epochNdcg20 = [0] * 2
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = args.batch
        steps = int(np.ceil(num / tstBat))
        for i in range(steps):
            st = i * tstBat
            ed = min((i+1) * tstBat, num)
            batIds = ids[st: ed]
            feed_dict = {}
            uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list = self.sampleTestBatch(batIds, self.handler.trnMat)
            feed_dict[self.uids] = uLocs
            feed_dict[self.iids] = iLocs
            feed_dict[self.is_train] = False
            feed_dict[self.sequence] = sequence
            feed_dict[self.mask] = mask
            feed_dict[self.uLocs_seq] = uLocs_seq
            feed_dict[self.keepRate] = 1.0
            preds = self.sess.run(self.preds, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            if(args.uid!=-1):
                print(preds[args.uid])
            if(args.test==True):
                hit5, ndcg5, hit10, ndcg10, hit15, ndcg15, hit20, ndcg20= self.calcRes(np.reshape(preds, [ed-st, args.testSize]), temTst, tstLocs)
            else:
                hit5, ndcg5, hit10, ndcg10, hit15, ndcg15, hit20, ndcg20=self.calcRes(np.reshape(preds, [ed-st, args.testSize]), val_list, tstLocs)
            epochHit5 += hit5
            epochNdcg5 += ndcg5
            epochHit10 += hit10
            epochNdcg10 += ndcg10
            epochHit15 += hit15
            epochNdcg15 += ndcg15
            epochHit20 += hit20
            epochNdcg20 += ndcg20
            log('Steps %d/%d: hit10 = %d, ndcg10 = %d' % (i, steps, hit10, ndcg10), save=False, oneline=True)
        ret = dict()
        ret['HR5'] = epochHit5 / num
        ret['NDCG5'] = epochNdcg5 / num
        ret['HR10'] = epochHit10 / num
        ret['NDCG10'] = epochNdcg10 / num
        ret['HR15'] = epochHit15 / num
        ret['NDCG15'] = epochNdcg15 / num
        ret['HR20'] = epochHit20 / num
        ret['NDCG20'] = epochNdcg20 / num
        print("epochNdcg5:{},epochHit5:{},epochNdcg10:{},epochHit10:{}".format(epochNdcg5/ num,epochHit5/ num,epochNdcg10/ num,epochHit10/ num))
        print("epochNdcg15:{},epochHit15:{},epochNdcg20:{},epochHit20:{}".format(epochNdcg15/ num,epochHit15/ num,epochNdcg20/ num,epochHit20/ num))
        return ret

    def calcRes(self, preds, temTst, tstLocs):
        hit10 = 0
        ndcg10 = 0
        hit1 = 0
        ndcg1 = 0
        hit5=0
        ndcg5=0
        hit20=0
        ndcg20=0
        hit15=0
        ndcg15=0
        for j in range(preds.shape[0]):
            predvals = list(zip(preds[j], tstLocs[j]))
            predvals.sort(key=lambda x: x[0], reverse=True)
            # 5
            k = list(map(lambda x: x[1], predvals[:5]))
            if temTst[j] in k:
                hit5 += 1
                ndcg5 += np.reciprocal(np.log2(k.index(temTst[j])+2))
            # 10
            k = list(map(lambda x: x[1], predvals[:args.k]))
            if temTst[j] in k:
                hit10 += 1
                ndcg10 += np.reciprocal(np.log2(k.index(temTst[j])+2))
            # 15
            k = list(map(lambda x: x[1], predvals[:15]))
            if temTst[j] in k:
                hit15 += 1
                ndcg15 += np.reciprocal(np.log2(k.index(temTst[j])+2))
            # 20
            k = list(map(lambda x: x[1], predvals[:20]))	
            if temTst[j] in k:
                hit20 += 1
                ndcg20 += np.reciprocal(np.log2(k.index(temTst[j])+2))	
        return hit5, ndcg5, hit10, ndcg10, hit15, ndcg15, hit20, ndcg20
    
    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        saver = tf.train.Saver()
        saver.save(self.sess, 'Models/' + args.save_path)
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, 'Models/' + args.load_model)
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')	
