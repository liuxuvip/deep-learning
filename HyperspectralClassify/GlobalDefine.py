run_version = 'version_3.9'

'''
Run History

1. version 3.8:
    This version the learn rate does not change, aim to see the best point to change learn rate point.
    It turns out that to change learn rate at epoch 10 and 30, not only will accelerate the study process,
    but decrease error as well.
    However, when epoch goes above 500, because of over-fitting, error goes up again
    
2. version 3.9:
    This version try to figure out the relation ship between D_loss and error.
    This test turns out that when D becomes better, it decrease over-fitting phenomenon clearly.
    
3. version 3.10:
    This version try to figure out how strong will D be to maximum the likelihood of the product of G.
    
'''

