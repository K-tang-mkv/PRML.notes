# Pytorch Autograd mechanics

### Graph retaining
If part of the autograd graph is shared between threads, i.e. run first part of forward single thread, then run second part in multiple threads, then the first part of graph is shared. In this case different threads execute grad() or backward() on the same graph might have issue of destroying the graph on the fly of one thread, and the other thread will crash in this case. Autograd will error out to the user similar to what call backward() twice with out retain_graph=True, and let the user know they should use retain_graph=True.
