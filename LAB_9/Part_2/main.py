from functions import *
from model import *
from utils import *

if __name__ == "__main__":
    vocab, lang, train_loader, dev_loader, test_loader = load_data('dataset/ptb.train.txt', 
                                                                   'dataset/ptb.valid.txt',
                                                                   'dataset/ptb.test.txt')
    
    ppl_replace = eval_ppl(lang, train_loader, dev_loader, test_loader, 0)
    
    ppl_dropout = eval_ppl(lang, train_loader, dev_loader, test_loader, 1)
    
    ppl_adam = eval_ppl(lang, train_loader, dev_loader, test_loader, 2)