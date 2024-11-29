# LigDockTailor
Customizing Docking Program Selection Through Ligand Feature Modeling

Please refer to 'config.txt' to configure the relevant environment before use

Usage:
1. Download /building/LigDockTailor_model.rar to the user-defined PATH and unzip it.
2. Download /example/LigDockTailor_main.py
3. Copy 'LigDockTailor_model.joblib' and 'LigDockTailor_main.py' to the working directory
4. Use the command in the working directory containing the files to be processed:

   python LigDockTailor_main.py [-h] [-im INPUT_MOL] [-in INPUT_NAME] [-oc OUTPUT_CSV_FILE] [-oz OUTPUT_ZIP_FILE] [-on OUTPUT_NAME_FILE]

  optional arguments:
  
    -h, --help            show this help message and exit
    
    -im INPUT_MOL, --input_mol INPUT_MOL
                          The molecular file to be processed must be in sdf format (RDkit/Open Babel is
                          recommended for format conversion).
                          
    -in INPUT_NAME, --input_name INPUT_NAME
                          User-defined identifiers for all molecules in the molecule file to be
                          processed, required to be a readable text file (XX.txt, XX.csv, etc.).
                          
    -oc OUTPUT_CSV_FILE, --output_csv_file OUTPUT_CSV_FILE
                          Output csv file containing classification information
                          
    -oz OUTPUT_ZIP_FILE, --output_zip_file OUTPUT_ZIP_FILE
                          Output zip compressed file containing sdf files of molecules classified into
                          various docking programs
                          
    -on OUTPUT_NAME_FILE, --output_name_file OUTPUT_NAME_FILE
                          Output file with the user-defined identifiers of all molecules after classification
