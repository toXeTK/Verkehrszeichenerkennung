import os
counter = 1
path = "C:/Users/kaise/OneDrive/TSE/Verkehrszeichenerkennung/projekt_hk_kt/"
checkpoint_filepath = path + '/chpt/' + f"{counter}"
os.makedirs(checkpoint_filepath, exist_ok=True)