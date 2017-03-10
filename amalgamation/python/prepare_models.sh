#!/usr/bin/env bash
script_name=$0

USAGE (){
    echo "  "
    echo "  "
    echo "Script downloads model from the MXNet Model Gallery "
    echo "  and prepares a combined JSON file containing the  "
    echo "  computation graph and weights."
    echo "Usage:"
    echo "${script_name} [-all] [-squeezenet] [-nin] [-caffenet] [-resnet] [-inceptionbn]"
    echo "  "
    echo "  "
}







prep_resnet_model(){
  echo "Preparing resnet18 model..."

  echo "    Downloading model from gallery..."
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/synset.txt
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/resnet/18-layers/resnet-18-0000.params
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/resnet/18-layers/resnet-18-symbol.json
}

prep_vgg_model(){
  echo "Preparing vgg-19 model..."

  echo "    Downloading model from gallery..."
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/synset.txt
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/vgg/vgg19-0000.params
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/vgg/vgg19-symbol.json
}

prep_nin_model(){
  echo "Preparing caffenet model..."

  echo "    Downloading model from gallery..."
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/synset.txt
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/nin/nin-0000.params
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/nin/nin-symbol.json
}

prep_caffenet_model(){
  echo "Preparing caffenet model..."

  echo "    Downloading model from gallery..."
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/synset.txt
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/caffenet/caffenet-symbol.json
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/caffenet/caffenet-0000.params
}

prep_squeezenet_model(){
  echo "Preparing squeezenet model..."

  echo "    Downloading model from gallery..."
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/synset.txt
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.0-symbol.json
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.0-0000.params
}

prep_inception_model(){
  echo "Preparing inceptionbn model..."

  echo "    Downloading inception model from gallery..."
  wget --no-check-certificate http://data.dmlc.ml/mxnet/models/imagenet/inception-bn.tar.gz
  tar -zxvf inception-bn.tar.gz
}


#
# Parse command-line arguments
#
while [ "$1" != "" ]; do
  case $1 in
    -help)
      USAGE
      exit 0;;
      -all | -inceptionbn | -squeezenet | -nin | -caffenet | -resnet | -vgg)
      TYPE=${1:1};;
  esac
  shift
done

if [ ! "$TYPE" ]; then
  echo "You must specify a test type"
  USAGE
  exit 0
fi


#
# Create temp dir
#
THIS_DIR=$(cd `dirname $0`; pwd)
TEMP_DIR="${THIS_DIR}/temp/"

if [[ ! -d "${TEMP_DIR}" ]]; then
  echo "${TEMP_DIR} doesn't exist, will create one";
  mkdir -p ${TEMP_DIR}
fi
cd ${TEMP_DIR}

#
# Prepare models
#
case $TYPE in
  all)
    echo "Preparing all models..."
    prep_nin_model
    prep_inception_model
    prep_squeezenet_model
    prep_resnet_model
    ;;
  nin)
    prep_nin_model
    ;;
  inceptionbn)
    prep_inception_model
    ;;
  squeezenet)
    prep_squeezenet_model
    ;;
  resnet)
    prep_resnet_model
    ;;
  vgg)
    prep_vgg_model
    ;;
  caffenet)
    prep_caffenet_model
    ;;
esac

echo "   Cleaning..."
sleep 2
cd ${THIS_DIR}
rm -rf ${TEMP_DIR}

echo "Done."
echo " "
echo "Contents for this dir:"
ls -ltrh
