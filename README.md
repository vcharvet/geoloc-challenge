# Geolocalisation Challenge


## Connection to cluster
(lame21)
$ ssh -L 8889:localhost:8889 user@137.194.192.179 




## Description of features
The database consists in a series of messages (probably mobile) that are emitted 
by a network of 159 antennas and received by clients.



rf(500, 'sqrt', 40) trained on PCA(0.95) gives 6.5 +- 0.1 on 5cv, (59min)
rf(500, 'sqrt', 40) trained on X_scale gives 6.9 +- 0.44 on 5cv (43min)

xgb(100, 10) trained on X_scale gives 6.9 +- 0.44 (39min)
xgb(100, 10) trained on PCA(0.95) gives  +-  (min)
q
