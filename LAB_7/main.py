from functions import *

if __name__ == "__main__":

    train_data, test_data = load_data()
    
    crf = define_crf()
    
    trn_feats, trn_label, tst_feats = set_data(train_data, test_data, 'baseline')    
    baseline_pred = train_and_predict(crf, trn_feats, trn_label, tst_feats)
    print("\nBASELINE RESULTS:")
    show_result(tst_feats, baseline_pred, test_data)
    
    trn_feats, trn_label, tst_feats = set_data(train_data, test_data, 'suffix')
    suffix_pred = train_and_predict(crf, trn_feats, trn_label, tst_feats)
    print("\nSUFFIX RESULTS:")
    show_result(tst_feats, suffix_pred, test_data)
    
    trn_feats, trn_label, tst_feats = set_data(train_data, test_data, 'tutorial')
    tutorial_pred = train_and_predict(crf, trn_feats, trn_label, tst_feats)
    print("\nALL TUTORIAL FEATURES RESULTS:")
    show_result(tst_feats, tutorial_pred, test_data)
    
    trn_feats, trn_label, tst_feats = set_data(train_data, test_data, 'window1')
    window1_pred = train_and_predict(crf, trn_feats, trn_label, tst_feats)
    print("\nINCREASING WINDOW1 RESULTS:")
    show_result(tst_feats, window1_pred, test_data)
    
    trn_feats, trn_label, tst_feats = set_data(train_data, test_data, 'window2')
    window2_pred = train_and_predict(crf, trn_feats, trn_label, tst_feats)
    print("\nINCREASING WINDOW2 RESULTS:\n")
    show_result(tst_feats, window2_pred, test_data)