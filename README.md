###

## Experiment Details:  

The objective of the experiment was to see what would happen 
if a the hidden-hidden weights of an fully trained MLP were 
randomly projected (using the guarantees provided by the 
Johnson-Lindenstrauss Lemma) and a new MLP with the same, 
albeit narrower architecture was created with the projected matrix 
and tested. As expected, the results were abysmal.   

Eplison values of 0.40 to 0.99 were used to project the matrix



#### Results: 

The fully trained MLP with 2 hidden layers yeided a test error of __1.8%__.   
The same MLP when initialized according to [1] and tested yielded an error of __90.2%__.  
A lot of the networks tested using the projected matrices yeilded an error __worse__ than __90.2%__ as can
be seen in the plot 

![alt text](https://github.com/vikkamath/randomProjections/blob/master/results.png, "Results")

















#### Bibliography: 
[1] Xavier Glorot, Yoshua Bengio: Understanding the difficulty of training deep feedforward neural networks, AISTATS 2010





