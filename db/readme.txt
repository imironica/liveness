short description

This directory contains images in raw format without under any further processing. There are 5105 client images and 7509 imposter images in total. Images from each subject are stored in a separate directory.

1. for evaluation
training set:
client   images: stored in the ClientRaw directory and indexed in client_train_raw.txt
imposter images: stored in the ImposterRaw directory and indexed in imposter_train_raw.txt

testing set:
client   images: stored in the ClientRaw directory and indexed in client_test_raw.txt
imposter images: stored in the ImposterRaw directory and indexed in imposter_test_raw.txt


2. The file name format of the indexed images: 

ID_galss_pos_session_picNo 

E.g. 0010_01_05_03_115.jpg
ID£º0001~0016 picture number
Glasses£º00~01
 	00£ºwith glasses
	01£ºno glasses
Pos£º   01~08 the location and light conditions of images
	01: up-down-rotate
        02: up-down-twist
        03: left-right-rotate
        04: left-right-twist
        05: close--window-open-lights
        07: open-window-open-lights
        08: open-widow-shut-lights
        08: still
Session£º01~03
picNo£ºpicture number


