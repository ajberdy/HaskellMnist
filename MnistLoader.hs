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

data Pixelgram = Pixel Float | Nothing deriving (Show)

activation :: Pixelgram -> Float
activation (Pixel a) = a

alignment :: Pixelgram -> Pixelgram -> Float
alignment (Pixel a) (Pixel b) = max 0 (1 - 2 * abs (a - b) / (a + b))

prior :: Pixelgram -> Float
prior (Pixel a) = 1 / 255

logprior :: Pixelgram -> Float
logprior = log . prior

imageSize :: Integer
imageSize = 28 * 28

datasetSize :: [V.Vector Int] -> Integer
datasetSize a = imageSize * fromIntegral (length a)

prominence :: Pixelgram -> [V.Vector Int] -> Float
prominence pixelgram images =
    sum $ map (V.sum . activations pixelgram) images

activations :: Pixelgram -> V.Vector Int -> V.Vector Float
activations pixelgram =
  V.map (alignment pixelgram . (Pixel . fromIntegral))
