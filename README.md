# RCMLRS
- Ramless - Cluster - Machine - Learning - Rust - Syntax

Machine learning framework in Rust, ramless, cluster compute
Very early trials of creations...

 TODO docs
 Notes
 - Must delete txt file not text inside
 - creation does not use buf so TODO use buf

Main idea Split ML models's into many smaller matrices unlike most that make one giant matrix.
Keep Tensor's in Storage when not applying operations, to apply operation load into Ram and store back.
Ie use your're storage similar to Ram and your're Ram like a CPU reister.
Using only storage would cause too much slowdowns, so to limit IO we can store only when not apply any operations.
To cluster compute matrices split are into chunks and share in a cluster.

This Machine Learning libary has the following activation functions, as implimations on RamTensor struct:

Impl RamTensor

- ReLU `max(0,x)`
- Leaky ReLU `max(ax,x)`
- Sigmoid `1/1+e^-x`
- Tanh `(e^x - e^-x) / (e^x - e^-x)`
- Softmax `
- Swish `
- GELU `
