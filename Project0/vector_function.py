def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    import numpy as np
    from scalar_function import scalar_function

    nfunc = np.vectorize(scalar_function)

    return(nfunc(x,y))