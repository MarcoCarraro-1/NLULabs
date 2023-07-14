from functions import *
from utils import *
from model import *

if __name__ == "__main__":
    
    PAD_TOKEN, train_raw, dev_raw, test_raw, y_train, y_dev, y_test = load_data('dataset/ATIS/train.json', 
                                                         'dataset/ATIS/test.json')
    lang = get_lang(train_raw, dev_raw, test_raw, y_train, y_dev, y_test, PAD_TOKEN)
    train_dataset, dev_dataset, test_dataset = get_dataset(train_raw, dev_raw, test_raw, lang)
    train_loader, dev_loader, test_loader = get_dataload(train_dataset, dev_dataset, 
                                                         test_dataset)
    
    f1_bidir, intent_acc_bidir = eval_f1_acc(lang, train_loader, dev_loader, test_loader, PAD_TOKEN, 0)
    
    f1_dropout, intent_acc_dropout = eval_f1_acc(lang, train_loader, dev_loader, test_loader, PAD_TOKEN, 1)