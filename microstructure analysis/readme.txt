Copyright (c) 2021 Jesper Oranje and Clemens Dransfeld
MIT License

@author: Jesper Oranje
This code has been writen for the authors master thesis at TUDelft which can be found in the TUDelft repository.

Input file for this code is a .csv file created by FIJI ImageJ particle analysis.
The FIJI ImageJ particle analysis pipeline and macro code is given and further explained in the authors Master thesis.  
The .csv file needs to contain the columns 'X','Y','Major','Minor','Angle'
    'X' & 'Y' are the pixel coordinates for every fibre centre location
    'Major' & 'Minor' are the axis length in pixels for the ellipse major and minor axis
    'Angle' is the angle in radians of the major axis relattive to the horizontal axis in the micrograph
 
Advised is to run this code on a cell by cell basis in spyder.
This can be done by selecting the cell and pressing Ctrl+Enter or by clicking on the "run current cell" icon in the toolbar.
