import MnistLoader
import Filtering

main :: IO ()
main = do
  labeled <- readIDXData' trainLabels trainImages
  let (labels, images) = unzip $ take 100 labeled
  -- print $ take 5 labels

  -- let pixel_activations = activations (Pixel 12) (head images)
  -- print $ zip $ pixel_activations [0..255]
  print "Success"
