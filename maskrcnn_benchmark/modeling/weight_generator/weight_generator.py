class Classifier(BasicModule):
    def __init__(self,nKall=64,nFeat=128*5*5, pre_weight_base, weight_generator_type='none'):
        super(Classifier,self).__init__()
        self.nKall=nKall
        self.nFeat=nFeat
        self.weight_generator_type=weight_generator_type
        
        weight_base=torch.FloatTensor(nKall,nFeat).normal_(
        0.0,np.sqrt(2.0/nFeat))
        self.weight_base=nn.Parameter(weight_base,requires_grad=True)
        ##load previous weights 


        self.bias=nn.Parameter(torch.FloatTensor(1).fill_(0),requires_grad=True)
        scale_cls=10.0
        self.scale_cls=nn.Parameter(torch.FloatTensor(1).fill_(scale_cls),requires_grad=True)
        
        if self.weight_generator_type=='none':
            #if type is none , then feature averaging is being used.
            #However,in this case the generator doesn't involve any learnable params ,thus doesn't require training
            self.favgblock=AvgBlock(nFeat)
        elif self.weight_generator_type=='attention_based':
            scale_att=10.0
            self.favgblock=AvgBlock(nFeat)
            self.attentionBlock=AttentionBlock(nFeat,nKall,scale_att=scale_att)
            
            self.wnLayerFavg=LinearDiag(nFeat)
            self.wnLayerWatt=LinearDiag(nFeat)
        else:
            raise ValueError('weight_generator_type is not supported!')
            
    def get_classification_weights(
        self,Kbase_ids,features_train=None,labels_train=None):
        """
        Args:
            Get the classification weights of the base and novel categories.
            Kbase_ids:[batch_size , nKbase],the indices of base categories that used
            features_train:[batch_size,num_train_examples(way*shot),nFeat]
            labels_train :[batch_size,num_train_examples,nKnovel(way)] one-hot of features_train
        
        return:
            cls_weights:[batch_size,nK,nFeat] 
        """
        import pdb; pdb.set_trace()
        #get the classification weights for the base categories
        batch_size,nKbase=Kbase_ids.size()
        weight_base=self.weight_base[Kbase_ids.contiguous().view(-1)]
        weight_base=weight_base.contiguous().view(batch_size,nKbase,-1)
        
        #if training data for novel categories are not provided,return only base_weight
        if features_train is None or labels_train is None:
            return weight_base
        
        #get classification weights for novel categories
        _,num_train_examples , num_channels=features_train.size()
        nKnovel=labels_train.size(2)
        
        #before do cosine similarity ,do L2 normalize
        features_train=F.normalize(features_train,p=2,dim=features_train.dim()-1,eps=1e-12)
        if self.weight_generator_type=='none':
            weight_novel=self.favgblock(features_train,labels_train)
            weight_novel=weight_novel.view(batch_size,nKnovel,num_channels)
        elif self.weight_generator_type=='attention_based':
            weight_novel_avg=self.favgblock(features_train,labels_train)
            weight_novel_avg=self.wnLayerFavg(weight_novel_avg.view(batch_size*nKnovel,num_channels))
            
            #do L2 for weighr_base
            weight_base_tmp=F.normalize(weight_base,p=2,dim=weight_base.dim()-1,eps=1e-12)
            
            weight_novel_att=self.attentionBlock(features_train,labels_train,weight_base_tmp,Kbase_ids)
            weight_novel_att=self.wnLayerWatt(weight_novel_att.view(batch_size*nKnovel,num_channels))
            import pdb; pdb.set_trace()
            weight_novel=weight_novel_avg+weight_novel_att
            weight_novel=weight_novel.view(batch_size,nKnovel,num_channels)
        else:
            raise ValueError('weight generator type is not supported!')
            
        #Concatenate the base and novel classification weights and return
        weight_both=torch.cat([weight_base,weight_novel],dim=1)#[batch_size ,nKbase+nKnovel , num_channel]
        
        return weight_both
    
    def apply_classification_weights(self,features,cls_weights):
        """
        Apply the classification weight vectors to the feature vectors
        Args:
            features:[batch_size,num_test_examples,num_channels]
            cls_weights:[batch_size,nK,num_channels]
        Return:
            cls_scores:[batch_size,num_test_examples(query set),nK]
        """
        #do L2 normalize
        features=F.normalize(features,p=2,dim=features.dim()-1,eps=1e-12)
        cls_weights=F.normalize(cls_weights,p=2,dim=cls_weights.dim()-1,eps=1e-12)
        cls_scores=self.scale_cls*torch.baddbmm(1.0,
                    self.bias.view(1,1,1),1.0,features,cls_weights.transpose(1,2))
        return cls_scores
    
    def forward(self,features_test,Kbase_ids,features_train=None,labels_train=None):
        """
        features_support: feature get from freezed FE

        Recognize on the test examples both base and novel categories.
        Args:
            features_test:[batch_size,num_test_examples(query set),num_channels]
            Kbase_ids:[batch_size,nKbase] , the indices of base categories that are being used.
            features_train:[batch_size,num_train_examples,num_channels]
            labels_train:[batch_size,num_train_examples,nKnovel]
            
        Return:
            cls_score:[batch_size,num_test_examples,nKbase+nKnovel]
    
        """
        import pdb; pdb.set_trace()
        cls_weights=self.get_classification_weights(
            Kbase_ids,features_train,labels_train)
        cls_scores=self.apply_classification_weights(features_test,cls_weights)
        return cls_scores    