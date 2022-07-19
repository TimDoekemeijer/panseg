"""
    GETLISTS
    Defs for getting lists of images and masks per study

    @author: tdoekemeijer
"""
import os

#CR
def getListOfCRImages(dirName):
    # create a list of file and sub directories
    listOfImages = os.listdir(dirName)
    allImages = list()
    # Iterate over all the entries
    for entry in listOfImages:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            # Looks only at directories with CR in name
            if 'CR' in entry:
                allImages = allImages + getListOfCRImages(fullPath)
        # Only append specific files to list
        if 'ADC600_0.nii.gz' in fullPath or 'ADC600_0_1.nii.gz' in fullPath and 'label' not in fullPath:
            allImages.append(fullPath)
    allImages.sort()
    return allImages

def getListOfCRMasks(dirName):
    # create a list of file and sub directories
    listOfMasks = os.listdir(dirName)
    allMasks = list()
    # Iterate over all the entries
    for entry in listOfMasks:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            # Looks only at directories with CR in name
            if 'CR' in entry:
                allMasks = allMasks + getListOfCRMasks(fullPath)
        # Only append specific files to list
        if 'ADC600_0-label.nii.gz' in fullPath or 'ADC600_0_1-label.nii.gz' in fullPath:
            allMasks.append(fullPath)
    allMasks.sort()
    return allMasks

#REPRO
def getListOfREPROImages(dirName):
    # create a list of file and sub directories
    listOfImages = os.listdir(dirName)
    allImages = list()
    # Iterate over all the entries
    for entry in listOfImages:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allImages = allImages + getListOfREPROImages(fullPath)
        # Only append specific files to list
        if 'ADC600_0.nii.gz' in fullPath or 'ADC600_0_1.nii.gz' in fullPath or 'ADC600_0_2.nii.gz' in fullPath and 'label' not in fullPath:
            allImages.append(fullPath)
    allImages.sort()
    return allImages

def getListOfREPROMasks(dirName):
    # create a list of file and sub directories
    listOfMasks = os.listdir(dirName)
    allMasks = list()
    # Iterate over all the entries
    for entry in listOfMasks:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allMasks = allMasks + getListOfREPROMasks(fullPath)
        # Only append specific files to list
        if 'ADC600_0-label' in fullPath or 'ADC600_0_1-label.nii.gz' in fullPath or 'ADC600_0_2-label.nii.gz' in fullPath:
            allMasks.append(fullPath)
    allMasks.sort()
    return allMasks

#MATRIX
def getListOfMATRIXImages(dirName):
    # create a list of file and sub directories
    listOfImages = os.listdir(dirName)
    allImages = list()
    # Iterate over all the entries
    for entry in listOfImages:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allImages = allImages + getListOfMATRIXImages(fullPath)
        # Only append specific files to list
        if 'ADC600' in fullPath and 'label' not in fullPath:
            allImages.append(fullPath)
    allImages.sort()
    return allImages

def getListOfMATRIXMasks(dirName):
    # create a list of file and sub directories
    listOfMasks = os.listdir(dirName)
    allMasks = list()
    # Iterate over all the entries
    for entry in listOfMasks:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allMasks = allMasks + getListOfMATRIXMasks(fullPath)
        # Only append specific files to list
        if 'label.nii.gz' in fullPath:
            allMasks.append(fullPath)
    allMasks.sort()
    return allMasks

#PREPROCESSED IMAGES
def getListOfPPImages(dirName):
    # create a list of file and sub directories
    listOfImages = os.listdir(dirName)
    allImages = list()
    # Iterate over all the entries
    for entry in listOfImages:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allImages = allImages + getListOfPPImages(fullPath)
        # Only append specific files to list
        if 'ADC600' in fullPath and 'label' not in fullPath:
            allImages.append(fullPath)
    allImages.sort()
    return allImages

def getListOfPPMasks(dirName):
    # create a list of file and sub directories
    listOfMasks = os.listdir(dirName)
    allMasks = list()
    # Iterate over all the entries
    for entry in listOfMasks:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allMasks = allMasks + getListOfPPMasks(fullPath)
        # Only append specific files to list
        if 'label.nii.gz' in fullPath:
            allMasks.append(fullPath)
    allMasks.sort()
    return allMasks

#FORM PAIRS
def get_filename_pairs(listOfImages, listOfMasks):
    filename_pairs = list(zip(listOfImages, listOfMasks))
    return filename_pairs