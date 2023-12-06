
import torch
import torch.nn as nn
from torch.autograd import Variable
from beam_search import CaptionGenerator
from data_utils import __normalize as normalize_values,  __EOS_TOKEN
from torchvision import transforms
# from torch.nn.functional import log_softmax
# import random
import torch.nn.functional as F 

class CaptionModel(nn.Module):
    def __init__(self, cnn, vocab, embed_size = 256, rnn_size = 256, num_layers=  2, share_embed_wts = False):
        super(CaptionModel, self).__init__()
        self.vocab = vocab
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        for param in cnn.parameters():
            param.requires_grad_(False)
        # dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
        # activation layers
        self.prelu = nn.ReLU()
        modules = list(cnn.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        
        self.img_embed = nn.Linear(cnn.fc.in_features, embed_size)
        self.lstm = nn.LSTM(embed_size, 
                            self.rnn_size, 
                            self.num_layers, dropout = 0.1
                            )
        self.b1 = nn.BatchNorm1d(rnn_size)
        self.classifier = nn.Linear(rnn_size, len(vocab))

        self.embed = nn.Embedding(len(self.vocab), embed_size)
        if share_embed_wts:
            self.embed.weight = self.classifier.weight
    
    def forward(self, imgs, captions, lens):
        features = self.cnn(imgs) # imgs => batch, channel, h, w
        features = features.view(features.size(0), -1)
        img_feats = self.img_embed(features).unsqueeze(0)
        caption_embed = self.embed(captions)
        caption_embed = torch.cat((img_feats, caption_embed),0)
        output, self.hidden = self.lstm(caption_embed)
        output = self.classifier(output)
        return output

    def sample(self, imgs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        """Samples captions for given image features."""
        output = []
        preproc = [ transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(**normalize_values)]
        if isinstance(imgs, torch.Tensor):
            preproc = [transforms.ToPILImage()] + preproc
        img_transform = transforms.Compose(preproc)
        imgs = img_transform(imgs)
        (h, c) = (torch.randn(self.num_layers, 1, self.rnn_size).cuda(), torch.randn(self.num_layers, 1, self.rnn_size).cuda())

        features = self.cnn(imgs.unsqueeze(0).cuda())
        features = features.view(features.size(0), -1)
        img_feats = self.img_embed(features).unsqueeze(0)
        inputs = img_feats
        for i in range(max_len):
            x, (h, c) = self.lstm(inputs, (h, c))
            x = self.classifier(x)
            x = x.squeeze(1)
            predict = x.argmax(dim=1)
            output.append(predict.item())
            inputs = self.embed(predict.unsqueeze(0))
        return [self.vocab[i] for i in output]

    
    def save_checkpoint(self, filename):
        torch.save({'embedder_dict': self.embed.state_dict(),
                    'rnn_dict': self.lstm.state_dict(),
                    'cnn_dict': self.cnn.state_dict(),
                    'classifier_dict': self.classifier.state_dict(),
                    'vocab': self.vocab,
                    'model': self},
                   filename)

    def load_checkpoint(self, filename):
        cpnt = torch.load(filename)
        if 'cnn_dict' in cpnt:
            self.cnn.load_state_dict(cpnt['cnn_dict'])
        self.embed.load_state_dict(cpnt['embedder_dict'])
        self.lstm.load_state_dict(cpnt['rnn_dict'])

        self.classifier.load_state_dict(cpnt['classifier_dict'])
    
    def finetune_cnn(self, allow=True):
        for p in self.cnn.parameters():
            p.requires_grad = allow
        for p in self.cnn.fc.parameters():
            p.requires_grad = True      