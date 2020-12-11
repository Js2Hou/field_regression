# This is the python implement for the paper "delta-Norm-based Robust Regression with Applications to Image Analysis".

## Introduction To The Experimenntal Dataset
- Shape of Dataset

Test_book : (32, 32, 3, 100)<br>
Test_glasses : (32, 32, 3, 100)<br>
Test_hand : (32, 32, 7, 100)<br>
Test_illumin : (32, 32, 3, 100)<br>
Test_sarf : (32, 32, 3, 100)<br>
Train_DAT : (32, 32, 6, 100)<br>
- Introduction

There are 100 people meaning 100 classes in both training dataset and testing datasets. Different datasets have different numbers of image for dividuals, e.g. 3 in Test_book, but 7 in Test_hand. The size of face image is (32, 32).