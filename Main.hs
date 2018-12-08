import MnistLoader

main :: IO ()
main = do
  labeled <- readIDXData' trainLabels trainImages
  let (labels, images) = unzip $ take 100 labeled
  let image1 = head images
  let px0 = Pixel 128
  print $ prominence px0 images
