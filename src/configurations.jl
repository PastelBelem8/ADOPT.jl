# ------------------------------------------------------------------------
#  Configurations
# ------------------------------------------------------------------------
export  DEPENDENCY_DIR,
        QHV_EXECUTABLE,
        QHV_MAX_DIM,
        QHV_TEMP_DIR,
        TEMP_DIR

# Folders
DEPENDENCY_DIR = "deps";
TEMP_DIR = tempdir();

# Algorithms -------------------------------------------------------------

# Quick HyperVolume
QHV_TEMP_DIR = mktempdir(TEMP_DIR);
QHV_EXECUTABLE = "$DEPENDENCY_DIR/QHV/d";
QHV_MAX_DIM = 15;
