import os

def get_greyboxes_paths(directory,sized=True,size=64,single=True,DMC=False,OC_version=False,ID=False): 
    
    # NEW VERSION FOR NEW GREBOXES NAMES

    # creates a list of .npy files with specific elements
    #takes a directory path

    #greyboxes are created with a specific string: 
    # greybox_v01_size64_26628261111219_00000.npy
    # greybox_v01_unsized_26628261111219_00000.npy
    # greyboxes_v01_size64_26628261111219.npy
    # greyboxes_v01_unsized_26628261111219.npy 
    #number is only added if it is a file with a single array

    #set "size=int" to get only files with resized arrays of this specific size
    #set "size=False" to get only files with non resized original arrays, overrides size
    #set "single=True" to get only files with a single array
    #set "DMC=26628261111219" to get only files of this DMC Scan, can be int or str
    #set OC_version to get only files of a specific OC_run
    #set ID to only get boxes with this specific ID
    #       allows "int" which will be automatically transformed to a five digit str (00001)
    #       allows "str" which NEED to be five digit
    #       allows "list" of only int or only str of the above formats
    #set "single=False" to get only files with multiple arrays. DEPRECIATED since AE func only takes single array files at the moment.

    IDlist=False
    if not ID==False:
        if type(ID)==str:
            IDstring=ID
        elif type(ID)==int:
            IDstring=f"{ID:05d}"
        elif type(ID)==list:
            IDlist=ID

            if type(IDlist[0])==int:
                IDlist=['{:05d}'.format(item) for item in IDlist]
        else:
            print("EROOR: wrong ID Input")
            IDstring=False

    endstring=".npy"
    sizestring="unsized"
    ID_startpos=-9
    ID_endpos=-4
    DMC_startpos=-18
    DMC_endpos=-4
    size_startpos=-26
    size_endpos=-19
    
    if OC_version==False:
        OC_version_len=0
    else:
        OC_version_len=len(OC_version)
    OC_version_startpos=size_startpos-OC_version_len-1
    OC_version_endpos=size_startpos-1

    if single==True:
        DMC_startpos=DMC_startpos-6
        DMC_endpos=DMC_endpos-6
        size_startpos=size_startpos-6
        size_endpos=size_endpos-6    
        OC_version_startpos=OC_version_startpos-6
        OC_version_endpos=OC_version_endpos-6
    
    if sized==True:
        size_startpos=size_startpos+1
        OC_version_startpos=OC_version_startpos+1
        OC_version_endpos=OC_version_endpos+1
        sizestring="size"+str(size)
    if IDlist==False:
        greyboxes_path_list = [os.path.join(root, name)
                    for root, dirs, files in os.walk(directory)
                    for name in files
                    if (name.endswith(endstring) and name[size_startpos:size_endpos]==sizestring)
                    if (OC_version==False or name[OC_version_startpos:OC_version_endpos]==OC_version)
                    if (DMC==False or name[DMC_startpos:DMC_endpos]==str(DMC))
                    if (ID==False or name[ID_startpos:ID_endpos]==IDstring)]
    else:
        greyboxes_path_list = [os.path.join(root, name)
                    for root, dirs, files in os.walk(directory)
                    for name in files
                    if (name.endswith(endstring) and name[size_startpos:size_endpos]==sizestring)
                    if (OC_version==False or name[OC_version_startpos:OC_version_endpos]==OC_version)
                    if (DMC==False or name[DMC_startpos:DMC_endpos]==str(DMC))
                    if (ID==False or (name[ID_startpos:ID_endpos] in IDlist))]       
    
    return greyboxes_path_list