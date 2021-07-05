import os
import matplotlib.pyplot as plt
import numpy as np
 
def save(path, ext='png', close=True, verbose=True):
    """ Save a figure from pyplot.
    Parameters
    ----------
    path : string
    The path (and filename, without the extension) to save the
    figure to.
    ext : string (default='png')
    The file extension. This must be supported by the active
    matplotlib backend (see matplotlib.backends module). Most
    backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
    Whether to close the figure after saving. If you want to save
    the figure multiple times (e.g., to multiple formats), you
    should NOT close it in between saves or you will have to
    re-plot it.
    verbose : boolean (default=True)
    Whether to print information about when and where the image
    has been saved.
    """
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'
        
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
            
    # The final path to save to
    savepath = os.path.join(directory, filename)
            
    if verbose:
        print("Saving figure to '%s'..." % savepath),
     
    # Actually save the figure
    plt.savefig(savepath)
    # Close it
    if close:
        plt.close()
        
    if verbose:
        print("Done") 
    return
    
def save_data(path,ext='npz',verbose=True,**variables):
    """ Save a data set from numpy as an archive in format npz
    The files can be loaded with result=np.load(path)
    result.files gives the array woth the names of the variables
    result['variable_name'] gives the corresponding variable
    Parameters
    ----------
    path : string
    The path (and filename, without the extension) to save the
    data to.
    ext : string (default='npz')
    The file extension. It should be npz
    variables: variables to save written as key word arguments
    verbose : boolean (default=True)
    Whether to print information about when and where the data
    has been saved.
    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'
        
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
            
    # The final path to save to
    savepath = os.path.join(directory, filename)
            
    if verbose:
        print("Saving data to '%s'..." % savepath),
     
    # Actually save the figure
    np.savez(savepath,**variables)

    if verbose:
        print("Done") 
    return