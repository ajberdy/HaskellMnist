module MnistLoader where

import           Data.IDX
import qualified Data.Vector.Unboxed as V

{-|
to download data:
training:
tcurl -OL http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -OL http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
testing:
curl -OL http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -OL http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gzip *ubyte.gz -d
|-}

trainLabels :: String
trainImages :: String
trainLabels = "data/t10k-labels-idx1-ubyte"
trainImages = "data/t10k-images-idx3-ubyte"

readIDXData' :: String -> String -> IO [(Int, V.Vector Int)]
readIDXData' train_labels train_images = do
    let maybe_idx_labels = decodeIDXLabelsFile train_labels
    let maybe_idx_data = decodeIDXFile train_images
    Just idx_labels <- maybe_idx_labels
    Just idx_data <- maybe_idx_data
    let maybe_labeled = labeledIntData idx_labels idx_data
    let Just labeled = maybe_labeled
    return labeled
