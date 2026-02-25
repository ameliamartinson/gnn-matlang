#!/bin/bash
# Apply Chebyshev Spectral Design to a target script

SCRIPT_FILE=$1

# Insert import
sed -i '' '/from libs.utils import/a\
from chebyshev_approx.cheb_utils import ChebyshevSpectralDesign\
' "$SCRIPT_FILE"

# Replace SpectralDesign with ChebyshevSpectralDesign
sed -i '' 's/transform = SpectralDesign(/transform = ChebyshevSpectralDesign(num_probes=10, cheb_degree=30, /g' "$SCRIPT_FILE"

# Remove processed data
DATASET_LINE=$(grep "root=" "$SCRIPT_FILE" | head -n 1)
if [[ $DATASET_LINE =~ root=\"([^\"]+)\" ]]; then
    DATASET_DIR=${BASH_REMATCH[1]}
    echo "Found dataset dir: $DATASET_DIR in $SCRIPT_FILE"
    rm -rf "$DATASET_DIR/processed"
fi

# Change default model to GNNML3
sed -i '' 's/model = .\+().to(device)/model = GNNML3().to(device)/g' "$SCRIPT_FILE"
echo "Updated $SCRIPT_FILE"
