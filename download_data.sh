mkdir -p ./data

wget -P ./data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget -P ./data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget -P ./data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget -P ./data http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

gunzip ./data/train-images-idx3-ubyte.gz
gunzip ./data/train-labels-idx1-ubyte.gz
gunzip ./data/t10k-images-idx3-ubyte.gz
gunzip ./data/t10k-labels-idx1-ubyte.gz
